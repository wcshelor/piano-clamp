"""Thin orchestration around the official vendored CLaMP 3 inference path."""

from __future__ import annotations

import json
import contextlib
import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .data import (
    AUDIO_EXTENSIONS,
    MANIFEST_FIELDS,
    OPTIONAL_MANIFEST_FIELDS,
    read_manifest,
    resolve_source_files,
    write_manifest_snapshot,
)
from .prompts import load_prompts
from .provenance import (
    build_data_payload,
    build_model_payload,
    build_prompt_payload,
    ensure_input_lock,
    sha256_file,
)
from .similarity import (
    as_embedding,
    build_similarity_rows,
    cosine_similarity_matrix,
    load_named_embeddings,
    write_similarity_table,
)


UPSTREAM_COMMIT = "9016d2b0c8d12d1aa79c2e0ab201e6822bdc83a8"
C2_CHECKPOINT = (
    "weights_clamp3_c2_h_size_768_t_model_FacebookAI_xlm-roberta-base_"
    "t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_"
    "s_layers_12_p_size_64_p_length_512.pth"
)
SAAS_CHECKPOINT = (
    "weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_"
    "t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_"
    "s_layers_12_p_size_64_p_length_512.pth"
)


class EmbeddingError(RuntimeError):
    """Raised when preprocessing or model inference cannot complete safely."""


def _run_dir(config: Mapping[str, Any]) -> Path:
    return Path(config["paths"]["run_dir"])


def initialize_run(config: Mapping[str, Any]) -> Path:
    """Create the documented run tree and write the resolved configuration."""

    run_dir = _run_dir(config)
    for relative in (
        "embeddings/scores",
        "embeddings/prompts",
        "embeddings/audio",
        "tables",
        "figures",
        "logs",
        "temp",
        "inputs",
    ):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.resolved.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False, allow_unicode=True)
    return run_dir


def _vendor_root(config: Mapping[str, Any]) -> Path:
    vendor = Path(config["paths"]["repository_root"]) / "vendor" / "clamp3"
    if not (vendor / "code" / "extract_clamp3.py").is_file():
        raise EmbeddingError(f"vendored CLaMP 3 checkout is incomplete: {vendor}")
    try:
        result = subprocess.run(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EmbeddingError(f"could not verify vendored CLaMP 3 commit: {exc}") from exc
    actual = result.stdout.strip()
    expected = str(config.get("clamp3", {}).get("upstream_commit", UPSTREAM_COMMIT))
    if actual != expected:
        raise EmbeddingError(f"CLaMP 3 commit mismatch: expected {expected}, found {actual}")
    return vendor


def _require_checkpoint(config: Mapping[str, Any], model_name: str) -> Path:
    checkpoint = Path(config["models"][model_name]["checkpoint"])
    if not checkpoint.is_file():
        if model_name == "saas":
            hint = "Run 'bash models/download_checkpoint.sh saas' explicitly if audio is needed."
        else:
            hint = "Restore the existing C2 checkpoint; this project never downloads it automatically."
        raise EmbeddingError(f"{model_name.upper()} checkpoint is unavailable: {checkpoint}. {hint}")
    expected_digest = config["models"][model_name].get("sha256")
    verify_digest = config.get("clamp3", {}).get("verify_checkpoint_sha256", False)
    if expected_digest and verify_digest:
        actual_digest = sha256_file(checkpoint)
        if actual_digest != expected_digest:
            raise EmbeddingError(
                f"{model_name.upper()} checkpoint SHA-256 mismatch: expected {expected_digest}, "
                f"found {actual_digest}"
            )
    return checkpoint


def _require_c2_wiring(config: Mapping[str, Any], vendor: Path) -> Path:
    checkpoint = _require_checkpoint(config, "c2")
    config_text = (vendor / "code" / "config.py").read_text(encoding="utf-8")
    if '"weights_clamp3_c2"' not in config_text:
        raise EmbeddingError("vendor/clamp3/code/config.py is not configured for CLaMP 3 C2")
    wired = vendor / "code" / C2_CHECKPOINT
    if not wired.is_file() or wired.resolve() != checkpoint.resolve():
        raise EmbeddingError(
            "the protected C2 symlink is missing or points at a different checkpoint: "
            f"{wired}"
        )
    return checkpoint


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    label: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{label}] $ {shlex.join(str(item) for item in command)}\n")
        log.flush()
        try:
            subprocess.run(
                [str(item) for item in command],
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                env={**os.environ, "PYTHONHASHSEED": "0"},
            )
        except subprocess.CalledProcessError as exc:
            raise EmbeddingError(
                f"{label} failed with exit status {exc.returncode}; see {log_path}"
            ) from exc


def _stage_sources(sources: Mapping[str, Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item_id, source in sources.items():
        target = destination / f"{item_id}{source.suffix.lower()}"
        target.symlink_to(source.resolve())


def _archive_upstream_logs(runtime: Path, destination: Path) -> None:
    """Keep upstream per-file diagnostics in the external run, not the submodule."""

    source = runtime / "logs"
    if not source.is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)
    for log_file in source.iterdir():
        if log_file.is_file():
            shutil.copy2(log_file, destination / log_file.name)


def _extract_official(
    model_inputs: Path,
    output_dir: Path,
    *,
    code_dir: Path,
    log_path: Path,
    label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_command(
        [sys.executable, "extract_clamp3.py", str(model_inputs), str(output_dir), "--get_global"],
        cwd=code_dir,
        log_path=log_path,
        label=label,
    )


def _check_outputs(output_dir: Path, item_ids: Sequence[str]) -> None:
    failures: list[str] = []
    for item_id in item_ids:
        output = output_dir / f"{item_id}.npy"
        if not output.is_file():
            failures.append(f"missing {output.name}")
            continue
        try:
            array = np.load(output, allow_pickle=False)
            if array.shape != (1, 768):
                failures.append(f"{output.name} has shape {array.shape}")
                continue
            as_embedding(array, label=str(output))
        except (OSError, ValueError) as exc:
            failures.append(f"{output.name}: {exc}")
    if failures:
        joined = "; ".join(failures[:10])
        raise EmbeddingError(f"CLaMP 3 did not produce valid (1, 768) outputs: {joined}")


def _preprocess_scores(
    sources: Mapping[str, Path],
    *,
    vendor: Path,
    workspace: Path,
    log_path: Path,
) -> Path:
    """Run upstream XML→ABC→interleaved-ABC and MIDI→MTF preprocessors."""

    xml_sources = {
        item_id: source
        for item_id, source in sources.items()
        if source.suffix.lower() in {".mxl", ".musicxml", ".xml"}
    }
    midi_sources = {
        item_id: source
        for item_id, source in sources.items()
        if source.suffix.lower() in {".mid", ".midi"}
    }
    model_inputs = workspace / "model_inputs"
    model_inputs.mkdir()

    if xml_sources:
        standard = workspace / "standard_abc"
        interleaved = workspace / "interleaved_abc"
        standard.mkdir()
        interleaved.mkdir()
        abc_dir = vendor / "preprocessing" / "abc"
        xml2abc = abc_dir / "utils" / "xml2abc.py"
        interleave_path = abc_dir / "batch_interleaved_abc.py"
        spec = importlib.util.spec_from_file_location("clamp3_interleaved_abc", interleave_path)
        if spec is None or spec.loader is None:
            raise EmbeddingError(f"cannot load official interleaved ABC preprocessor: {interleave_path}")
        interleave = importlib.util.module_from_spec(spec)
        previous_bytecode_setting = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            spec.loader.exec_module(interleave)
        finally:
            sys.dont_write_bytecode = previous_bytecode_setting
        for item_id, source in xml_sources.items():
            standard_file = standard / f"{item_id}.abc"
            command = [sys.executable, str(xml2abc), "-d", "8", "-x", str(source)]
            result = subprocess.run(command, cwd=abc_dir, capture_output=True, text=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[MusicXML to standard ABC] $ {shlex.join(command)}\n")
                log.write(result.stderr)
            if result.returncode != 0 or not result.stdout.strip():
                raise EmbeddingError(
                    f"official MusicXML preprocessing produced no standard ABC for {item_id}; "
                    f"see {log_path}"
                )
            standard_file.write_text(result.stdout, encoding="utf-8")
            with log_path.open("a", encoding="utf-8") as log:
                with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                    interleave.abc_pipeline(standard_file, standard, interleaved)
            generated = interleaved / f"{item_id}.abc"
            if not generated.is_file() or generated.stat().st_size == 0:
                raise EmbeddingError(
                    f"official interleaved ABC preprocessing failed for {item_id}; see {log_path}"
                )
            (model_inputs / generated.name).symlink_to(generated)

    if midi_sources:
        mtf = workspace / "mtf"
        mtf.mkdir()
        midi_path = vendor / "preprocessing" / "midi" / "batch_midi2mtf.py"
        spec = importlib.util.spec_from_file_location("clamp3_midi2mtf", midi_path)
        if spec is None or spec.loader is None:
            raise EmbeddingError(f"cannot load official MIDI preprocessor: {midi_path}")
        midi2mtf = importlib.util.module_from_spec(spec)
        previous_bytecode_setting = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            spec.loader.exec_module(midi2mtf)
        finally:
            sys.dont_write_bytecode = previous_bytecode_setting
        for item_id, source in midi_sources.items():
            generated = mtf / f"{item_id}.mtf"
            try:
                generated.write_text(midi2mtf.load_midi(source, True), encoding="utf-8")
            except Exception as exc:
                raise EmbeddingError(f"official MIDI preprocessing failed for {item_id}: {exc}") from exc
            (model_inputs / generated.name).symlink_to(generated)
    return model_inputs


def embed_scores(config: Mapping[str, Any]) -> Path:
    """Embed manifest score passages using the protected C2 configuration."""

    vendor = _vendor_root(config)
    _require_c2_wiring(config, vendor)
    rows = read_manifest(config["paths"]["manifest"])
    sources = resolve_source_files(rows, config["paths"]["data_root"])
    run_dir = initialize_run(config)
    ensure_input_lock(run_dir, "score_data", build_data_payload(rows, sources))
    ensure_input_lock(run_dir, "score_model", build_model_payload(config, "c2"))
    write_manifest_snapshot(rows, run_dir / "manifest.snapshot.csv")
    output_dir = run_dir / "embeddings" / "scores"
    log_path = run_dir / "logs" / "embed_scores.log"
    with tempfile.TemporaryDirectory(prefix="scores-", dir=run_dir / "temp") as temporary:
        model_inputs = _preprocess_scores(
            sources,
            vendor=vendor,
            workspace=Path(temporary),
            log_path=log_path,
        )
        _extract_official(
            model_inputs,
            output_dir,
            code_dir=vendor / "code",
            log_path=log_path,
            label="CLaMP 3 C2 symbolic embedding",
        )
    _check_outputs(output_dir, [row["passage_id"] for row in rows])
    return output_dir


def embed_prompts(config: Mapping[str, Any]) -> Path:
    """Embed configured text prompts in the same C2 space as score passages."""

    vendor = _vendor_root(config)
    _require_c2_wiring(config, vendor)
    prompts = load_prompts(config["prompt_files"])
    run_dir = initialize_run(config)
    ensure_input_lock(run_dir, "prompt_data", build_prompt_payload(prompts))
    ensure_input_lock(run_dir, "prompt_model", build_model_payload(config, "c2"))
    output_dir = run_dir / "embeddings" / "prompts"
    log_path = run_dir / "logs" / "embed_prompts.log"
    with tempfile.TemporaryDirectory(prefix="prompts-", dir=run_dir / "temp") as temporary:
        model_inputs = Path(temporary) / "text"
        model_inputs.mkdir()
        for prompt in prompts:
            (model_inputs / f"{prompt['prompt_id']}.txt").write_text(
                prompt["prompt_text"] + "\n", encoding="utf-8"
            )
        _extract_official(
            model_inputs,
            output_dir,
            code_dir=vendor / "code",
            log_path=log_path,
            label="CLaMP 3 C2 text embedding",
        )
    _check_outputs(output_dir, [prompt["prompt_id"] for prompt in prompts])
    return output_dir


def score_prompts(config: Mapping[str, Any]) -> Path:
    """Write all score–prompt cosine similarities in the C2 space."""

    rows = read_manifest(config["paths"]["manifest"])
    sources = resolve_source_files(rows, config["paths"]["data_root"])
    prompts = load_prompts(config["prompt_files"])
    run_dir = initialize_run(config)
    ensure_input_lock(run_dir, "score_data", build_data_payload(rows, sources))
    ensure_input_lock(run_dir, "score_model", build_model_payload(config, "c2"))
    ensure_input_lock(run_dir, "prompt_data", build_prompt_payload(prompts))
    ensure_input_lock(run_dir, "prompt_model", build_model_payload(config, "c2"))
    write_manifest_snapshot(rows, run_dir / "manifest.snapshot.csv")
    score_vectors = load_named_embeddings(
        run_dir / "embeddings" / "scores", [row["passage_id"] for row in rows]
    )
    prompt_vectors = load_named_embeddings(
        run_dir / "embeddings" / "prompts", [prompt["prompt_id"] for prompt in prompts]
    )
    values = cosine_similarity_matrix(score_vectors, prompt_vectors)
    output = run_dir / "tables" / "prompt_similarity.csv"
    write_similarity_table(build_similarity_rows(rows, prompts, values), output)
    return output


def _make_saas_code_directory(vendor: Path, checkpoint: Path, destination: Path) -> Path:
    """Create an isolated SAAS inference config without changing vendored C2 files."""

    destination.mkdir(parents=True)
    for name in ("extract_clamp3.py", "utils.py", "config.py"):
        shutil.copy2(vendor / "code" / name, destination / name)
    config_path = destination / "config.py"
    config_text = config_path.read_text(encoding="utf-8")
    if '"weights_clamp3_c2"' not in config_text:
        raise EmbeddingError("cannot derive isolated SAAS config from the protected C2 config")
    config_path.write_text(
        config_text.replace('"weights_clamp3_c2"', '"weights_clamp3_saas"', 1),
        encoding="utf-8",
    )
    (destination / SAAS_CHECKPOINT).symlink_to(checkpoint.resolve())
    return destination


def embed_audio(config: Mapping[str, Any]) -> Path:
    """Optionally embed audio with SAAS; never downloads the SAAS checkpoint."""

    vendor = _vendor_root(config)
    checkpoint = _require_checkpoint(config, "saas")
    audio_manifest = config["paths"].get("audio_manifest")
    if not audio_manifest:
        raise EmbeddingError("paths.audio_manifest must be configured for embed-audio")
    rows = read_manifest(audio_manifest, allowed_extensions=AUDIO_EXTENSIONS)
    sources = resolve_source_files(rows, config["paths"]["data_root"])
    run_dir = initialize_run(config)
    ensure_input_lock(run_dir, "audio_data", build_data_payload(rows, sources))
    ensure_input_lock(run_dir, "audio_model", build_model_payload(config, "saas"))
    write_manifest_snapshot(rows, run_dir / "audio_manifest.snapshot.csv")
    output_dir = run_dir / "embeddings" / "audio"
    log_path = run_dir / "logs" / "embed_audio.log"
    with tempfile.TemporaryDirectory(prefix="audio-", dir=run_dir / "temp") as temporary:
        workspace = Path(temporary)
        staged = workspace / "audio"
        mert = workspace / "mert"
        _stage_sources(sources, staged)
        audio_config = config.get("audio", {})
        mert_model = str(audio_config.get("mert_model", "m-a-p/MERT-v1-95M"))
        audio_dir = vendor / "preprocessing" / "audio"
        runtime = workspace / "audio_runtime"
        runtime.mkdir()
        _run_command(
            [
                sys.executable,
                str(audio_dir / "extract_mert.py"),
                "--input_path",
                str(staged),
                "--output_path",
                str(mert),
                "--model_path",
                mert_model,
                "--mean_features",
            ],
            cwd=runtime,
            log_path=log_path,
            label="MERT audio preprocessing",
        )
        _archive_upstream_logs(runtime, log_path.parent / "upstream_audio")
        missing = [item_id for item_id in sources if not (mert / f"{item_id}.npy").is_file()]
        if missing:
            raise EmbeddingError(f"MERT preprocessing failed for: {', '.join(missing)}; see {log_path}")
        runtime_code = _make_saas_code_directory(vendor, checkpoint, workspace / "saas_code")
        _extract_official(
            mert,
            output_dir,
            code_dir=runtime_code,
            log_path=log_path,
            label="CLaMP 3 SAAS audio embedding",
        )
    _check_outputs(output_dir, [row["passage_id"] for row in rows])
    return output_dir


def make_run_manifest(config: Mapping[str, Any]) -> Path:
    """Record model identities and generated artifacts without hashing checkpoints."""

    run_dir = initialize_run(config)
    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir)
        if (
            path.is_file()
            and path.name != "run_manifest.json"
            and (not relative.parts or relative.parts[0] != "temp")
        ):
            artifacts.append(
                {
                    "path": str(relative),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    models = {}
    for name in ("c2", "saas"):
        checkpoint = Path(config["models"][name]["checkpoint"])
        models[name] = {
            "checkpoint": str(checkpoint),
            "available": checkpoint.is_file(),
            "size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
            "configured_sha256": config["models"][name].get("sha256"),
        }
    document = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clamp3_upstream_commit": config.get("clamp3", {}).get("upstream_commit", UPSTREAM_COMMIT),
        "models": models,
        "artifacts": artifacts,
        "manifest_fields": list(MANIFEST_FIELDS),
        "recognized_optional_manifest_fields": list(OPTIONAL_MANIFEST_FIELDS),
        "geometry_warning": "C2 and SAAS outputs are different checkpoint spaces and must not be mixed.",
    }
    output = run_dir / "run_manifest.json"
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return output
