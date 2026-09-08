"""Public adapter around the official directory-oriented CLaMP 3 pipeline.

Upstream does not expose a stable Python embedding class. Its supported public
surface is ``clamp3_embd.py`` plus the modality helpers in ``utils.py``; those
helpers preprocess directories and invoke ``code/extract_clamp3.py``. This
adapter preserves those exact preprocessors while providing item-level errors,
configured checkpoints, deterministic staging, and one model process per call.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .embeddings import C2_CHECKPOINT, SAAS_CHECKPOINT, _preprocess_scores
from .embedding_io import sha256_path


DISCOVERED_API = {
    "entrypoint": "vendor/clamp3/clamp3_embd.py <input_dir> <output_dir> --get_global",
    "text": "utils.extract_txt_features -> code/extract_clamp3.py (.txt)",
    "symbolic_musicxml": "utils.extract_xml_features -> XML/MXL -> ABC -> interleaved ABC",
    "symbolic_midi": "utils.extract_mid_features -> MIDI -> M3-compatible MTF",
    "audio": "utils.extract_audio_features -> MERT .npy -> code/extract_clamp3.py",
    "global_shape": [1, 768],
    "note": "extract_clamp3.py is a script with module-level argparse/model loading, not an import-safe API",
}


class ClampBackendError(RuntimeError):
    """Raised when configured official preprocessing or inference cannot run."""


def resolve_device(requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ClampBackendError("device must be auto, cpu, or cuda")
    try:
        import torch
    except ImportError as exc:
        if requested == "cuda":
            raise ClampBackendError("CUDA was requested but PyTorch is unavailable") from exc
        return "cpu"
    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise ClampBackendError("CUDA was requested, but torch.cuda.is_available() is false")
    return "cuda" if available and requested != "cpu" else "cpu"


def missing_runtime_modules(modality: str) -> list[str]:
    required = ["torch", "transformers", "accelerate", "samplings"]
    if modality in {"symbolic", "all"}:
        required.append("abctoolkit")
    if modality in {"audio", "all"}:
        required.extend(["torchaudio", "mido"])
    return sorted(name for name in set(required) if importlib.util.find_spec(name) is None)


class ClampBackend:
    """Run the pinned upstream code against an explicit checkpoint."""

    def __init__(self, config: Mapping[str, Any], *, model_space: str = "c2") -> None:
        if model_space not in {"c2", "saas"}:
            raise ClampBackendError("model_space must be c2 or saas")
        self.config = dict(config)
        self.model_space = model_space
        self.vendor = Path(str(config["vendor_root"]))
        checkpoint_key = "checkpoint_path" if model_space == "c2" else "saas_checkpoint_path"
        self.checkpoint = Path(str(config[checkpoint_key]))
        self.device = resolve_device(str(config.get("device", "auto")))
        self.config["resolved_device"] = self.device
        if not (self.vendor / "code" / "extract_clamp3.py").is_file():
            raise ClampBackendError(f"CLaMP checkout is incomplete: {self.vendor}")
        if not self.checkpoint.is_file():
            extra = (
                "The symbolic C2 checkpoint must be restored at the configured path."
                if model_space == "c2"
                else "Audio requires a separately trained SAAS checkpoint; it is never downloaded automatically."
            )
            raise ClampBackendError(f"checkpoint is unavailable: {self.checkpoint}. {extra}")
        expected_hash = (
            config.get("checkpoint_sha256")
            if model_space == "c2"
            else config.get("saas_checkpoint_sha256")
        )
        if expected_hash and config.get("verify_checkpoint_sha256", True):
            actual_hash = sha256_path(self.checkpoint)
            if actual_hash != str(expected_hash):
                raise ClampBackendError(
                    f"{model_space.upper()} checkpoint SHA-256 mismatch: expected {expected_hash}, found {actual_hash}"
                )

    @property
    def checkpoint_name(self) -> str:
        return C2_CHECKPOINT if self.model_space == "c2" else SAAS_CHECKPOINT

    def _runtime_code(self, destination: Path) -> Path:
        destination.mkdir(parents=True, exist_ok=False)
        for name in ("extract_clamp3.py", "utils.py", "config.py"):
            shutil.copy2(self.vendor / "code" / name, destination / name)
        config_path = destination / "config.py"
        text = config_path.read_text(encoding="utf-8")
        if self.model_space == "saas":
            if '"weights_clamp3_c2"' not in text:
                raise ClampBackendError("cannot derive SAAS runtime from the pinned upstream config")
            text = text.replace('"weights_clamp3_c2"', '"weights_clamp3_saas"', 1)
            config_path.write_text(text, encoding="utf-8")
        if self.config.get("low_memory_checkpoint_loading", True):
            extractor = destination / "extract_clamp3.py"
            extractor_text = extractor.read_text(encoding="utf-8")
            load_call = 'torch.load(checkpoint_path, map_location="cpu", weights_only=True)'
            if load_call not in extractor_text:
                raise ClampBackendError("pinned extractor checkpoint-load call was not found")
            extractor_text = extractor_text.replace(
                load_call,
                'torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)',
                1,
            )
            state_call = "model.load_state_dict(checkpoint['model'])"
            if state_call not in extractor_text:
                raise ClampBackendError("pinned extractor state-dict call was not found")
            tokenizer_call = "tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)"
            if tokenizer_call in extractor_text:
                extractor_text = extractor_text.replace(
                    tokenizer_call,
                    "tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME, local_files_only=True)",
                    1,
                )
            extractor.write_text(extractor_text, encoding="utf-8")
        utils_path = destination / "utils.py"
        utils_text = utils_path.read_text(encoding="utf-8")
        model_call = "self.text_model = AutoModel.from_pretrained(text_model_name) # Load the text model"
        if model_call in utils_text:
            utils_text = utils_text.replace(
                model_call,
                "self.text_model = AutoModel.from_pretrained(text_model_name, local_files_only=True) # Load the text model",
                1,
            )
            utils_path.write_text(utils_text, encoding="utf-8")
        (destination / self.checkpoint_name).symlink_to(self.checkpoint.resolve())
        return destination

    def _environment(self) -> dict[str, str]:
        environment = {**os.environ, "PYTHONHASHSEED": "0", "TOKENIZERS_PARALLELISM": "false"}
        if self.device == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
            environment["ACCELERATE_USE_CPU"] = "true"
        return environment

    def probe_runtime(self, *, workspace: Path, log_path: Path, modality: str) -> dict[str, Any]:
        if modality not in {"text", "symbolic", "audio"}:
            raise ClampBackendError("runtime probe modality must be text, symbolic, or audio")
        runtime = self._runtime_code(workspace / f"{self.model_space}_runtime_probe")
        probe_script = runtime / "probe_runtime.py"
        required = ["torch", "transformers", "accelerate"]
        if modality in {"text", "symbolic"}:
            required.extend(["sklearn", "scipy"])
        if modality == "symbolic":
            required.append("abctoolkit")
        if modality == "audio":
            required.extend(["torchaudio", "mido"])
        probe_script.write_text(
            textwrap.dedent(
                f"""
                import importlib
                import json
                import platform
                import sys
                import traceback

                result = {{
                    "python_executable": sys.executable,
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    "modules": {required!r},
                    "imports": {{}},
                }}
                ok = True
                for name in result["modules"]:
                    entry = {{"ok": False}}
                    try:
                        module = importlib.import_module(name)
                        entry["ok"] = True
                        entry["file"] = getattr(module, "__file__", None)
                        entry["version"] = getattr(module, "__version__", None)
                    except Exception as exc:
                        ok = False
                        entry["error_type"] = type(exc).__name__
                        entry["error"] = str(exc)
                        entry["traceback"] = traceback.format_exc()
                    result["imports"][name] = entry
                result["ok"] = ok
                print(json.dumps(result, indent=2))
                sys.exit(0 if ok else 1)
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        probe_path = workspace / f"{modality}_runtime_probe.json"
        command = [sys.executable, str(probe_script)]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[CLaMP 3 {self.model_space.upper()} runtime probe] $ {shlex.join(command)}\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=runtime,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=self._environment(),
            )
            payload = completed.stdout
            log.write(payload)
            log.flush()
        try:
            report = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ClampBackendError(
                f"CLaMP runtime probe produced unreadable output; see {log_path}"
            ) from exc
        probe_path.parent.mkdir(parents=True, exist_ok=True)
        probe_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if completed.returncode != 0:
            failing = next(
                (
                    (name, details)
                    for name, details in report.get("imports", {}).items()
                    if not details.get("ok", False)
                ),
                None,
            )
            if failing is None:
                raise ClampBackendError(f"CLaMP runtime probe failed unexpectedly; see {log_path}")
            name, details = failing
            message = str(details.get("error", "")).strip()
            extra = ""
            if "GLIBCXX_" in message:
                extra = " The runtime is missing a required libstdc++ symbol; rebuild or switch to an environment with a newer C++ runtime."
            raise ClampBackendError(
                f"CLaMP runtime probe failed while importing {name}: {message}; see {log_path}.{extra}"
            )
        return report

    def _run(self, command: list[str], *, cwd: Path, log_path: Path, label: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{label}] $ {shlex.join(command)}\n")
            log.flush()
            try:
                subprocess.run(
                    command,
                    cwd=cwd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    env=self._environment(),
                )
            except subprocess.CalledProcessError as exc:
                tail = ""
                try:
                    tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
                except OSError:
                    pass
                hint = ""
                if "out of memory" in tail.casefold() or exc.returncode in {-9, 137}:
                    hint = (
                        " The process was killed during model/checkpoint allocation; request more job memory "
                        "(or more GPU memory when using CUDA)."
                    )
                raise ClampBackendError(
                    f"{label} failed with exit status {exc.returncode}; see {log_path}.{hint}"
                ) from exc

    def _infer(self, inputs: Path, output: Path, *, workspace: Path, log_path: Path) -> None:
        runtime = self._runtime_code(workspace / f"{self.model_space}_runtime")
        output.mkdir(parents=True, exist_ok=True)
        self._run(
            [sys.executable, "extract_clamp3.py", str(inputs), str(output), "--get_global"],
            cwd=runtime,
            log_path=log_path,
            label=f"CLaMP 3 {self.model_space.upper()} global embedding",
        )

    @staticmethod
    def _collect(
        output: Path, item_ids: list[str], errors: dict[str, str]
    ) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        vectors: dict[str, np.ndarray] = {}
        for item_id in item_ids:
            if item_id in errors:
                continue
            path = output / f"{item_id}.npy"
            if not path.is_file():
                errors[item_id] = "CLaMP did not produce an output; inspect the stage log"
                continue
            try:
                array = np.load(path, allow_pickle=False)
            except (OSError, ValueError) as exc:
                errors[item_id] = f"cannot load CLaMP output: {exc}"
                continue
            if array.shape != (1, 768) or not np.isfinite(array).all():
                errors[item_id] = f"invalid CLaMP output shape or values: {array.shape}"
                continue
            vectors[item_id] = array[0].astype(np.float32, copy=False)
        return vectors, errors

    def embed_texts(
        self,
        texts: Mapping[str, str],
        *,
        workspace: Path,
        log_path: Path,
    ) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        missing = missing_runtime_modules("text")
        if missing:
            raise ClampBackendError(f"missing CLaMP runtime modules: {', '.join(missing)}")
        inputs = workspace / "text_inputs"
        outputs = workspace / "text_outputs"
        inputs.mkdir(parents=True)
        for item_id, text in texts.items():
            (inputs / f"{item_id}.txt").write_text(text + "\n", encoding="utf-8")
        self._infer(inputs, outputs, workspace=workspace, log_path=log_path)
        return self._collect(outputs, list(texts), {})

    def embed_symbolic(
        self,
        sources: Mapping[str, Path],
        *,
        workspace: Path,
        log_path: Path,
    ) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        missing = missing_runtime_modules("symbolic")
        if missing:
            raise ClampBackendError(f"missing CLaMP runtime modules: {', '.join(missing)}")
        inputs = workspace / "symbolic_inputs"
        inputs.mkdir(parents=True)
        errors: dict[str, str] = {}
        for item_id, source in sources.items():
            item_workspace = workspace / "preprocess" / item_id
            item_workspace.mkdir(parents=True)
            try:
                generated_dir = _preprocess_scores(
                    {item_id: Path(source)},
                    vendor=self.vendor,
                    workspace=item_workspace,
                    log_path=log_path,
                )
                generated = next(generated_dir.iterdir())
                (inputs / generated.name).symlink_to(generated.resolve())
            except Exception as exc:  # upstream converters raise several exception types
                errors[item_id] = f"symbolic preprocessing failed: {exc}"
        if any(inputs.iterdir()):
            outputs = workspace / "symbolic_outputs"
            self._infer(inputs, outputs, workspace=workspace, log_path=log_path)
        else:
            outputs = workspace / "symbolic_outputs"
        return self._collect(outputs, list(sources), errors)

    def embed_audio_files(
        self,
        sources: Mapping[str, Path],
        *,
        workspace: Path,
        log_path: Path,
    ) -> tuple[dict[str, np.ndarray], dict[str, str]]:
        if self.model_space != "saas":
            raise ClampBackendError("audio embeddings require the separately trained SAAS model space")
        missing = missing_runtime_modules("audio")
        if missing:
            raise ClampBackendError(f"missing CLaMP audio runtime modules: {', '.join(missing)}")
        staged = workspace / "audio_inputs"
        mert = workspace / "mert_features"
        outputs = workspace / "audio_outputs"
        staged.mkdir(parents=True)
        for item_id, source in sources.items():
            (staged / f"{item_id}{source.suffix.lower()}").symlink_to(Path(source).resolve())
        runtime = workspace / "mert_runtime"
        runtime.mkdir()
        model = str(self.config.get("mert_model", "m-a-p/MERT-v1-95M"))
        self._run(
            [
                sys.executable,
                str(self.vendor / "preprocessing" / "audio" / "extract_mert.py"),
                "--input_path",
                str(staged),
                "--output_path",
                str(mert),
                "--model_path",
                model,
                "--mean_features",
            ],
            cwd=runtime,
            log_path=log_path,
            label="official MERT audio preprocessing",
        )
        errors = {
            item_id: "MERT preprocessing did not produce an output"
            for item_id in sources
            if not (mert / f"{item_id}.npy").is_file()
        }
        if mert.is_dir() and any(mert.iterdir()):
            self._infer(mert, outputs, workspace=workspace, log_path=log_path)
        return self._collect(outputs, list(sources), errors)


def inspect_backend(config: Mapping[str, Any]) -> dict[str, Any]:
    vendor = Path(str(config["vendor_root"]))
    checkpoint = Path(str(config["checkpoint_path"]))
    saas = Path(str(config["saas_checkpoint_path"]))
    try:
        commit = subprocess.run(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {
        "api": DISCOVERED_API,
        "vendor_root": str(vendor),
        "vendor_commit": commit,
        "c2_checkpoint": str(checkpoint),
        "c2_checkpoint_available": checkpoint.is_file(),
        "saas_checkpoint": str(saas),
        "saas_checkpoint_available": saas.is_file(),
        "requested_device": config.get("device", "auto"),
        "resolved_device": resolve_device(str(config.get("device", "auto"))),
        "missing_text_modules": missing_runtime_modules("text"),
        "missing_symbolic_modules": missing_runtime_modules("symbolic"),
        "missing_audio_modules": missing_runtime_modules("audio"),
    }
