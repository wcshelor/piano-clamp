#!/usr/bin/env python3
"""Generate provenance-rich score or aligned-audio passages and embed them."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.clamp_backend import ClampBackend, ClampBackendError, resolve_device  # noqa: E402
from piano_clamp.data_loading import (  # noqa: E402
    filter_manifest_rows,
    load_pipeline_config,
    read_corpus_manifest,
)
from piano_clamp.embedding_io import (  # noqa: E402
    align_success_rows,
    content_hash,
    guard_bundle,
    sha256_path,
    write_embedding_bundle,
)
from piano_clamp.passage_artifacts import write_failed_passage_rows  # noqa: E402
from piano_clamp.embedding_store import snapshot_written_bundle  # noqa: E402
from piano_clamp.metadata import build_run_metadata, make_run_id, stage_log_path  # noqa: E402
from piano_clamp.power import add_caffeinate_arguments, maybe_caffeinate  # noqa: E402
from piano_clamp.passage_generation import (  # noqa: E402
    PASSAGE_FIELDS,
    extract_aligned_audio_passage,
    generate_passage_records,
    load_alignment_times,
    materialize_symbolic_passage,
    parse_bar_ranges,
    rights_authorized,
)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)


def _write_chunk_manifest(
    chunk_dir: Path,
    *,
    chunk_index: int,
    total_chunks: int,
    digests: list[str],
    digest_sources: dict[str, Path],
    digest_to_passage_ids: dict[str, list[str]],
) -> Path:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "unique_digest_count": len(digests),
        "digests": [
            {
                "digest": digest,
                "source_path": str(digest_sources[digest]),
                "passage_ids": list(digest_to_passage_ids.get(digest, [])),
            }
            for digest in digests
        ],
    }
    path = chunk_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _format_seconds(value: float) -> str:
    total = max(0, int(round(value)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


def _status_counts(status_map: dict[str, dict[str, str]]) -> tuple[int, int, int]:
    success = sum(1 for item in status_map.values() if item.get("status") == "success")
    failed = sum(1 for item in status_map.values() if item.get("status") == "failed")
    pending = sum(1 for item in status_map.values() if item.get("status") == "pending")
    return success, failed, pending


def _run_key(
    *,
    modality: str,
    source_material: str,
    modes: list[str],
    stride_bars: int | None,
    user_ranges: list[tuple[int, int]],
    composer: str | None,
    work_id: str | None,
    recording_id: str | None,
    limit: int | None,
    records: list[dict[str, object]],
) -> str:
    return content_hash(
        {
            "modality": modality,
            "source_material": source_material,
            "modes": list(modes),
            "stride_bars": stride_bars,
            "user_ranges": [list(item) for item in user_ranges],
            "composer": composer,
            "work_id": work_id,
            "recording_id": recording_id,
            "limit": limit,
            "passage_ids": [row["passage_id"] for row in records],
        }
    )


def _initial_resume_state(
    *,
    run_key: str,
    total_records: int,
    modality: str,
    source_material: str,
    modes: list[str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_key": run_key,
        "modality": modality,
        "source_material": source_material,
        "modes": list(modes),
        "total_records": total_records,
        "record_status": {},
    }


def _adopt_subset_resume_state(
    state: dict[str, object],
    *,
    run_key: str,
    total_records: int,
    modality: str,
    source_material: str,
    modes: list[str],
    expected_passage_ids: set[str],
) -> dict[str, object]:
    record_status = {
        str(key): value
        for key, value in dict(state.get("record_status", {})).items()
        if str(key) in expected_passage_ids and isinstance(value, dict)
    }
    return {
        "schema_version": 1,
        "run_key": run_key,
        "modality": modality,
        "source_material": source_material,
        "modes": list(modes),
        "total_records": total_records,
        "record_status": record_status,
    }


def _load_or_initialize_resume_state(
    *,
    resume_dir: Path,
    run_key: str,
    total_records: int,
    modality: str,
    source_material: str,
    modes: list[str],
    expected_passage_ids: set[str],
    reset: bool,
) -> dict[str, object]:
    state_path = resume_dir / "state.json"
    if reset and resume_dir.exists():
        for child in sorted(resume_dir.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    resume_dir.mkdir(parents=True, exist_ok=True)
    if not state_path.is_file():
        state = _initial_resume_state(
            run_key=run_key,
            total_records=total_records,
            modality=modality,
            source_material=source_material,
            modes=modes,
        )
        _atomic_json(state_path, state)
        return state
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("run_key") != run_key:
        saved_ids = {
            str(key)
            for key, value in dict(state.get("record_status", {})).items()
            if isinstance(value, dict)
        }
        if expected_passage_ids and expected_passage_ids.issubset(saved_ids):
            adopted = _adopt_subset_resume_state(
                state,
                run_key=run_key,
                total_records=total_records,
                modality=modality,
                source_material=source_material,
                modes=modes,
                expected_passage_ids=expected_passage_ids,
            )
            _atomic_json(state_path, adopted)
            print(
                "Reusing a superset passage resume cache and trimming it to the current run.",
                flush=True,
            )
            return adopted
        raise ValueError(
            "existing passage resume cache does not match this run; pass --reset-resume to discard it"
        )
    return state


def _save_resume_state(resume_dir: Path, state: dict[str, object]) -> None:
    _atomic_json(resume_dir / "state.json", state)


def _prepared_cache_path(
    prepared_dir: Path,
    passage_id: str,
    saved_entry: dict[str, str] | None,
) -> Path | None:
    prepared_value = str((saved_entry or {}).get("prepared_path", "")).strip()
    if prepared_value:
        candidate = prepared_dir.parent / prepared_value
        if candidate.is_file():
            return candidate
    matches = sorted(prepared_dir.glob(f"{passage_id}.*"))
    return matches[0] if matches else None


def _method_spec(name: str) -> tuple[str, int | None]:
    mapping = {
        "complete_movement": ("complete_movement", None),
        "complete": ("complete_movement", None),
        "fixed_4_bars": ("fixed_bar_window", 4),
        "4": ("fixed_bar_window", 4),
        "fixed_8_bars": ("fixed_bar_window", 8),
        "8": ("fixed_bar_window", 8),
        "fixed_16_bars": ("fixed_bar_window", 16),
        "16": ("fixed_bar_window", 16),
        "phrase": ("phrase", None),
        "cadence": ("cadence", None),
        "user_range": ("user_range", None),
        "user": ("user_range", None),
    }
    if name not in mapping:
        raise ValueError(f"unknown passage method: {name}")
    return mapping[name]


def _source_material(args: argparse.Namespace) -> str:
    if args.source_material != "auto":
        return str(args.source_material)
    return "audio" if args.modality == "audio" else "score"


def _output_directory(output_root: Path, source_material: str) -> Path:
    output = output_root / "passages"
    if source_material == "score":
        return output
    return output / source_material


def run(args: argparse.Namespace) -> dict[str, object]:
    config = load_pipeline_config(args.config, device=args.device)
    source_material = _source_material(args)
    if args.modality == "audio" and source_material != "audio":
        raise ValueError("--modality audio requires --source-material audio")
    if args.modality == "symbolic" and source_material not in {"score", "performance_midi"}:
        raise ValueError("--modality symbolic requires --source-material score or performance_midi")
    stage_name = "passages_symbolic" if source_material == "score" else f"passages_{source_material}"
    run_id = make_run_id(stage_name, config)
    manifest_rows = filter_manifest_rows(
        read_corpus_manifest(config["corpus_manifest"]),
        composer=args.composer,
        work_id=args.work_id,
    )
    if args.recording_id:
        manifest_rows = [row for row in manifest_rows if row["recording_id"] == args.recording_id]
    modes = args.mode or list(config.get("passage_methods", ["complete_movement"]))
    user_ranges = parse_bar_ranges(args.bar_range or [])
    records = []
    seen = set()
    mode_counts: Counter[str] = Counter()
    for mode in modes:
        method, window = _method_spec(mode)
        if method == "user_range" and not user_ranges:
            raise ValueError("user_range mode requires at least one --bar-range START-END")
        effective_stride = args.stride_bars
        if method == "fixed_bar_window" and window is not None and args.overlap_half and effective_stride is None:
            effective_stride = max(1, window // 2)
        generated = generate_passage_records(
            manifest_rows,
            corpus_root=config["corpus_root"],
            method=method,
            window_bars=window,
            stride_bars=effective_stride,
            user_ranges=user_ranges,
        )
        for row in generated:
            if row["passage_id"] not in seen:
                seen.add(row["passage_id"])
                records.append(row)
                mode_counts[str(row.get("passage_generation_method", mode))] += 1
    records.sort(key=lambda row: row["passage_id"])
    if args.limit is not None:
        records = records[: args.limit]

    modality = args.modality
    log_path = stage_log_path(
        config,
        f"embed_{stage_name}",
        run_id=run_id,
    )
    output = _output_directory(Path(config["output_root"]), source_material)
    guard_bundle(
        output,
        ("passage_embeddings.npy", "passage_embeddings.csv", "metadata.json"),
        force=args.force,
    )
    run_key = _run_key(
        modality=modality,
        source_material=source_material,
        modes=list(modes),
        stride_bars=args.stride_bars,
        user_ranges=list(user_ranges),
        composer=args.composer,
        work_id=args.work_id,
        recording_id=args.recording_id,
        limit=args.limit,
        records=records,
    )
    resume_dir = output / ".resume"
    vector_cache = resume_dir / "vectors"
    prepared_cache = resume_dir / "prepared"
    failed_rows_path = resume_dir / "failed_passages.csv"
    state = _load_or_initialize_resume_state(
        resume_dir=resume_dir,
        run_key=run_key,
        total_records=len(records),
        modality=modality,
        source_material=source_material,
        modes=list(modes),
        expected_passage_ids={str(record["passage_id"]) for record in records},
        reset=args.reset_resume,
    )
    status_map: dict[str, dict[str, str]] = {
        str(key): {str(k): str(v) for k, v in value.items()}
        for key, value in dict(state.get("record_status", {})).items()
        if isinstance(value, dict)
    }
    resumed_success, resumed_failed, resumed_pending = _status_counts(status_map)
    temporary_root = Path(config["temporary_root"])
    temporary_root.mkdir(parents=True, exist_ok=True)
    events = load_alignment_times(config["alignment_root"])
    allowed_rights = list(config.get("authorized_rights_statuses", []))
    global_error = ""
    pending_digests: list[str] = []
    digest_sources: dict[str, Path] = {}
    digest_to_passage_ids: dict[str, list[str]] = {}
    resolved_before_embed = 0
    prepared_cache_hits = 0
    started_at = time.monotonic()
    print(
        "Planned passage counts by method: "
        + ", ".join(f"{name}={count}" for name, count in sorted(mode_counts.items())),
        flush=True,
    )
    print(
        f"Resume cache status: {resumed_success} success, {resumed_failed} failed, "
        f"{resumed_pending} pending across {len(status_map)} cached records.",
        flush=True,
    )
    print(
        f"Preparing/materializing {len(records)} passage records for source_material={source_material}...",
        flush=True,
    )
    runtime_probe_path = resume_dir / f"{source_material}_runtime_probe.json"
    with tempfile.TemporaryDirectory(prefix=f"passages-{source_material}-", dir=temporary_root) as temporary:
        workspace = Path(temporary)
        model_space = "c2" if modality == "symbolic" else "saas"
        backend = ClampBackend(config, model_space=model_space)
        config["resolved_device"] = backend.device
        print(
            f"Running {model_space.upper()} runtime probe before passage preparation on device={backend.device}...",
            flush=True,
        )
        probe = backend.probe_runtime(
            workspace=workspace / "runtime_probe",
            log_path=log_path,
            modality="symbolic" if modality == "symbolic" else "audio",
        )
        runtime_probe_path.write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
        print(
            f"Runtime probe passed; diagnostic report written to {runtime_probe_path}.",
            flush=True,
        )
        materialized = workspace / "materialized"
        for record_index, record in enumerate(records, start=1):
            record["embedding_modality"] = source_material
            record["source_material"] = source_material
            passage_id = str(record["passage_id"])
            saved_entry = status_map.get(passage_id, {})
            cached_prepared = _prepared_cache_path(prepared_cache, passage_id, saved_entry)
            skip_queue = False
            try:
                if cached_prepared is not None:
                    path = cached_prepared
                    digest = str(saved_entry.get("content_hash", "")).strip() or sha256_path(path)
                    prepared_cache_hits += 1
                else:
                    if modality == "symbolic":
                        temp_path = materialize_symbolic_passage(
                            record,
                            corpus_root=config["corpus_root"],
                            destination=materialized,
                            source_material=source_material,
                            events=events,
                        )
                    else:
                        if not rights_authorized(record.get("rights_status", ""), allowed_rights):
                            record["status"] = "unauthorized"
                            record["error_message"] = "audio rights_status is not explicitly authorized"
                            continue
                        temp_path, onset, offset = extract_aligned_audio_passage(
                            record,
                            corpus_root=config["corpus_root"],
                            events=events,
                            destination=materialized,
                            sample_rate=int(config.get("audio_sample_rate", 24000)),
                        )
                        record["onset_seconds"] = onset
                        record["offset_seconds"] = offset
                        record["source_path"] = str(record.get("audio_path", ""))
                    prepared_cache.mkdir(parents=True, exist_ok=True)
                    path = prepared_cache / f"{passage_id}{temp_path.suffix.lower()}"
                    if not path.is_file():
                        shutil.copy2(temp_path, path)
                    digest = sha256_path(path)
                if source_material == "score":
                    record["source_path"] = str(record.get("score_path") or record.get("midi_path", ""))
                elif source_material == "performance_midi":
                    record["source_path"] = str(record.get("performance_midi_path", ""))
                else:
                    record["source_path"] = str(record.get("audio_path", ""))
                record["content_hash"] = digest
                digest_to_passage_ids.setdefault(digest, []).append(passage_id)
                saved_entry["prepared_path"] = str(path.relative_to(resume_dir))
                cached_vector = vector_cache / f"{digest}.npy"
                if cached_vector.is_file():
                    resolved_before_embed += 1
                    status_map[passage_id] = {
                        "status": "success",
                        "error_message": "",
                        "content_hash": digest,
                        "prepared_path": str(path.relative_to(resume_dir)),
                    }
                    skip_queue = True
                else:
                    if digest not in digest_sources:
                        digest_sources[digest] = path
                    if digest not in pending_digests:
                        pending_digests.append(digest)
                    status_map[passage_id] = {
                        "status": "pending",
                        "error_message": "",
                        "content_hash": digest,
                        "prepared_path": str(path.relative_to(resume_dir)),
                    }
            except Exception as exc:
                record["status"] = "failed"
                record["error_message"] = str(exc)
                status_map[passage_id] = {
                    "status": "failed",
                    "error_message": str(exc),
                    "content_hash": str(record.get("content_hash", "")),
                    "prepared_path": str(saved_entry.get("prepared_path", "")),
                }
            if (
                record_index == 1
                or record_index == len(records)
                or record_index % args.status_every == 0
            ):
                prepared_elapsed = time.monotonic() - started_at
                resolved = sum(1 for item in status_map.values() if item.get("status") in {"success", "failed"})
                rate = record_index / prepared_elapsed if prepared_elapsed > 0 else 0.0
                remaining = len(records) - record_index
                eta = remaining / rate if rate > 0 else 0.0
                print(
                    f"Preparation progress: {record_index}/{len(records)} records scanned; "
                    f"{len(digest_sources)} unique uncached contents queued; "
                    f"{prepared_cache_hits} prepared-cache hits; "
                    f"{resolved} records already resolved; "
                    f"elapsed {_format_seconds(prepared_elapsed)}; "
                    f"ETA {_format_seconds(eta)}.",
                    flush=True,
                )
            if skip_queue:
                continue
        state["record_status"] = status_map
        _save_resume_state(resume_dir, state)
        write_failed_passage_rows(
            failed_rows_path,
            records=records,
            status_map=status_map,
        )

        print(
            f"Prepared {len(records)} passage records; {len(digest_sources)} unique uncached contents remain; "
            f"{prepared_cache_hits} prepared-cache hits; "
            f"{resolved_before_embed} vector-cache hits.",
            flush=True,
        )

        if pending_digests:
            try:
                chunk_size = args.chunk_size or len(pending_digests)
                total_chunks = math.ceil(len(pending_digests) / chunk_size)
                for chunk_index, start in enumerate(range(0, len(pending_digests), chunk_size), start=1):
                    chunk_started_at = time.monotonic()
                    chunk = pending_digests[start : start + chunk_size]
                    chunk_manifest = _write_chunk_manifest(
                        resume_dir / "chunks" / f"chunk_{chunk_index:04d}",
                        chunk_index=chunk_index,
                        total_chunks=total_chunks,
                        digests=chunk,
                        digest_sources=digest_sources,
                        digest_to_passage_ids=digest_to_passage_ids,
                    )
                    print(
                        f"Embedding chunk {chunk_index}/{total_chunks} "
                        f"({start + 1}-{start + len(chunk)} of {len(pending_digests)} unique pending contents). "
                        f"Manifest: {chunk_manifest}",
                        flush=True,
                    )
                    chunk_workspace = workspace / "backend" / f"chunk_{chunk_index:04d}"
                    chunk_sources = {digest: digest_sources[digest] for digest in chunk}
                    if modality == "symbolic":
                        vectors, errors = backend.embed_symbolic(
                            chunk_sources,
                            workspace=chunk_workspace,
                            log_path=log_path,
                        )
                    else:
                        vectors, errors = backend.embed_audio_files(
                            chunk_sources,
                            workspace=chunk_workspace,
                            log_path=log_path,
                        )
                    for digest, vector in vectors.items():
                        _atomic_npy(vector_cache / f"{digest}.npy", vector)
                        for passage_id in digest_to_passage_ids.get(digest, []):
                            status_map[passage_id] = {
                                "status": "success",
                                "error_message": "",
                                "content_hash": digest,
                            }
                    for digest, message in errors.items():
                        for passage_id in digest_to_passage_ids.get(digest, []):
                            status_map[passage_id] = {
                                "status": "failed",
                                "error_message": message,
                                "content_hash": digest,
                            }
                    state["record_status"] = status_map
                    _save_resume_state(resume_dir, state)
                    write_failed_passage_rows(
                        failed_rows_path,
                        records=records,
                        status_map=status_map,
                    )
                    completed = sum(1 for item in status_map.values() if item.get("status") == "success")
                    failed = sum(1 for item in status_map.values() if item.get("status") == "failed")
                    chunk_elapsed = time.monotonic() - chunk_started_at
                    total_elapsed = time.monotonic() - started_at
                    chunks_done = chunk_index
                    avg_chunk = total_elapsed / chunks_done if chunks_done > 0 else 0.0
                    chunks_left = total_chunks - chunks_done
                    eta = chunks_left * avg_chunk
                    print(
                        f"Progress: {completed + failed}/{len(records)} records resolved "
                        f"({completed} success, {failed} failed). "
                        f"Chunk time {_format_seconds(chunk_elapsed)}; "
                        f"total elapsed {_format_seconds(total_elapsed)}; "
                        f"ETA {_format_seconds(eta)}.",
                        flush=True,
                    )
            except ClampBackendError as exc:
                global_error = str(exc)
                state["record_status"] = status_map
                _save_resume_state(resume_dir, state)
                write_failed_passage_rows(
                    failed_rows_path,
                    records=records,
                    status_map=status_map,
                )
                raise
        else:
            model_space = "c2" if modality == "symbolic" else "saas"
            config["resolved_device"] = resolve_device(str(config["device"]))

    vectors: dict[str, np.ndarray] = {}
    for record in records:
        passage_id = str(record["passage_id"])
        digest = str(record.get("content_hash", ""))
        if digest:
            cached_vector = vector_cache / f"{digest}.npy"
            if cached_vector.is_file():
                vectors[passage_id] = np.load(cached_vector, allow_pickle=False)
        saved = status_map.get(passage_id)
        if saved:
            record["status"] = saved.get("status", record.get("status", "pending"))
            record["error_message"] = saved.get("error_message", record.get("error_message", ""))
    matrix, aligned = align_success_rows(records, vectors, id_field="passage_id")
    checkpoint_key = "checkpoint_path" if modality == "symbolic" else "saas_checkpoint_path"
    metadata = build_run_metadata(
        config,
        stage=stage_name,
        checkpoint_path=config[checkpoint_key],
        model_space=model_space,
        run_id=run_id,
        batch_size=int(config["batch_size_symbolic"] if modality == "symbolic" else config["batch_size_audio"]),
        input_manifest_path=config["corpus_manifest"],
        extra={
            "log_path": str(log_path),
            "passage_methods": modes,
            "source_material": source_material,
            "fixed_window_stride_bars": args.stride_bars,
            "fixed_window_overlap_half": bool(args.overlap_half),
            "user_bar_ranges": [list(item) for item in user_ranges],
            "unique_content_count": len(digest_to_passage_ids),
            "audio_alignment_policy": "Audio passages require explicit score-measure onset/offset events; unaligned ranges are never inferred.",
            "global_error": global_error or None,
            "resume_directory": str(resume_dir),
            "runtime_probe_path": str(runtime_probe_path),
            "chunk_size": args.chunk_size or len(pending_digests) or 0,
            "prepared_cache_directory": str(prepared_cache),
            "prepared_cache_hits": prepared_cache_hits,
        },
    )
    paths = write_embedding_bundle(
        output,
        matrix,
        aligned,
        matrix_filename="passage_embeddings.npy",
        table_filename="passage_embeddings.csv",
        fields=PASSAGE_FIELDS,
        id_field="passage_id",
        metadata=metadata,
        force=args.force,
        normalize=bool(config["normalize_embeddings"]),
    )
    store = snapshot_written_bundle(paths, config)
    failed = sum(row["status"] == "failed" for row in aligned)
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "store": {key: str(value) if isinstance(value, Path) else value for key, value in store.items()},
        "log_path": str(log_path),
        "failed_rows_path": str(failed_rows_path),
        "embedded": matrix.shape[0],
        "failed": failed,
        "unsupported_unavailable_or_unauthorized": len(records) - matrix.shape[0] - failed,
        "global_error": global_error,
        "resume_directory": str(resume_dir),
        "source_material": source_material,
        "resolved_before_embed": resolved_before_embed,
        "prepared_cache_hits": prepared_cache_hits,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--modality", choices=("symbolic", "audio"), default="symbolic")
    parser.add_argument(
        "--source-material",
        choices=("auto", "score", "performance_midi", "audio"),
        default="auto",
        help="material to embed: score notation, performance MIDI, or aligned audio",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=("complete", "4", "8", "16", "phrase", "cadence", "user"),
        help="repeat to combine passage-generation modes; defaults to the config list",
    )
    parser.add_argument("--bar-range", action="append", help="START-END for user mode")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--composer")
    parser.add_argument("--work-id")
    parser.add_argument("--recording-id")
    parser.add_argument(
        "--stride-bars",
        type=int,
        help="step size for fixed-bar windows; defaults to the window size, so smaller values create overlap",
    )
    parser.add_argument(
        "--overlap-half",
        action="store_true",
        help="for fixed 4/8/16-bar windows, default each mode to 50%% overlap unless --stride-bars is explicit",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="number of unique uncached passage contents to embed per checkpointed chunk",
    )
    parser.add_argument(
        "--status-every",
        type=int,
        default=25,
        help="emit preparation progress every N records before chunked embedding starts",
    )
    parser.add_argument("--reset-resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    add_caffeinate_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.stride_bars is not None and args.stride_bars < 1:
        parser.error("--stride-bars must be positive")
    if args.overlap_half and args.stride_bars is not None:
        parser.error("--overlap-half cannot be combined with --stride-bars")
    if args.chunk_size is not None and args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.status_every is not None and args.status_every < 1:
        parser.error("--status-every must be positive")
    try:
        with maybe_caffeinate(args.caffeinate, flags=args.caffeinate_flags):
            result = run(args)
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")
    print(result)
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
