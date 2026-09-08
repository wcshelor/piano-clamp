#!/usr/bin/env python3
"""Embed only present and explicitly authorized local recordings with SAAS."""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.clamp_backend import ClampBackend, ClampBackendError, resolve_device  # noqa: E402
from piano_clamp.data_loading import (  # noqa: E402
    filter_manifest_rows,
    load_pipeline_config,
    read_corpus_manifest,
    select_recording_records,
)
from piano_clamp.embedding_io import (  # noqa: E402
    align_success_rows,
    guard_bundle,
    sha256_path,
    write_embedding_bundle,
)
from piano_clamp.embedding_store import snapshot_written_bundle  # noqa: E402
from piano_clamp.metadata import build_run_metadata, make_run_id, stage_log_path  # noqa: E402
from piano_clamp.passage_generation import (  # noqa: E402
    chunk_audio,
    load_audio,
    rights_authorized,
    write_wav,
)
from piano_clamp.power import add_caffeinate_arguments, maybe_caffeinate  # noqa: E402


RECORDING_FIELDS = (
    "recording_id",
    "composer",
    "composition_id",
    "work_id",
    "movement_id",
    "performer",
    "audio_path",
    "rights_status",
    "audio_origin",
    "content_hash",
    "duration_seconds",
    "chunk_count",
    "aggregation_method",
    "embedding_row",
    "embedding_dimension",
    "status",
    "error_message",
)

CHUNK_FIELDS = (
    "chunk_id",
    "recording_id",
    "chunk_index",
    "start_seconds",
    "end_seconds",
    "embedding_row",
    "embedding_dimension",
    "status",
    "error_message",
)


def run(args: argparse.Namespace) -> dict[str, object]:
    config = load_pipeline_config(args.config, device=args.device)
    run_id = make_run_id("audio", config)
    log_path = stage_log_path(config, "embed_audio", run_id=run_id)
    manifest_rows = filter_manifest_rows(
        read_corpus_manifest(config["corpus_manifest"]),
        composer=args.composer,
        work_id=args.work_id,
    )
    records = select_recording_records(manifest_rows, config["corpus_root"])
    if args.recording_id:
        records = [row for row in records if row["recording_id"] == args.recording_id]
    if args.limit is not None:
        records = records[: args.limit]
    output = Path(config["output_root"]) / "audio"
    chunks_output = output / "chunks"
    guard_bundle(
        output,
        ("recording_embeddings.npy", "recording_embeddings.csv", "metadata.json"),
        force=args.force,
    )
    guard_bundle(
        chunks_output,
        ("chunk_embeddings.npy", "chunk_embeddings.csv", "metadata.json"),
        force=args.force,
    )

    allowed = list(config.get("authorized_rights_statuses", []))
    sample_rate = int(config.get("audio_sample_rate", 24000))
    chunk_seconds = float(config.get("audio_chunk_seconds", 30.0))
    overlap_seconds = float(config.get("audio_chunk_overlap_seconds", 0.0))
    temporary_root = Path(config["temporary_root"])
    temporary_root.mkdir(parents=True, exist_ok=True)
    chunk_records: list[dict[str, object]] = []
    chunk_sources: dict[str, Path] = {}
    recording_chunks: dict[str, list[str]] = defaultdict(list)
    global_error = ""
    with tempfile.TemporaryDirectory(prefix="audio-", dir=temporary_root) as temporary:
        workspace = Path(temporary)
        wav_dir = workspace / "chunks"
        for record in records:
            source = record.pop("resolved_audio_path", None)
            record.update(
                {
                    "content_hash": "",
                    "duration_seconds": "",
                    "chunk_count": 0,
                    "aggregation_method": str(config.get("audio_aggregation", "mean")),
                }
            )
            if record["status"] != "pending" or not source:
                continue
            if not rights_authorized(record["rights_status"], allowed):
                record["status"] = "unauthorized"
                record["error_message"] = (
                    "local audio was not embedded because rights_status is not in authorized_rights_statuses"
                )
                continue
            record["content_hash"] = sha256_path(source)
            try:
                waveform, rate = load_audio(
                    source,
                    sample_rate=sample_rate,
                    mono=bool(config.get("audio_mono", True)),
                )
                chunks = chunk_audio(
                    waveform,
                    rate,
                    chunk_seconds=chunk_seconds,
                    overlap_seconds=overlap_seconds,
                )
                record["duration_seconds"] = waveform.shape[0] / rate
                record["chunk_count"] = len(chunks)
                for index, (start, end, chunk) in enumerate(chunks):
                    chunk_id = f"{record['recording_id']}__chunk_{index:05d}"
                    target = wav_dir / f"{chunk_id}.wav"
                    write_wav(target, chunk, rate)
                    chunk_sources[chunk_id] = target
                    recording_chunks[record["recording_id"]].append(chunk_id)
                    chunk_records.append(
                        {
                            "chunk_id": chunk_id,
                            "recording_id": record["recording_id"],
                            "chunk_index": index,
                            "start_seconds": start,
                            "end_seconds": end,
                            "status": "pending",
                            "error_message": "",
                        }
                    )
            except Exception as exc:
                record["status"] = "failed"
                record["error_message"] = str(exc)

        chunk_vectors: dict[str, np.ndarray] = {}
        chunk_errors: dict[str, str] = {}
        if chunk_sources:
            try:
                backend = ClampBackend(config, model_space="saas")
                config["resolved_device"] = backend.device
                chunk_vectors, chunk_errors = backend.embed_audio_files(
                    chunk_sources,
                    workspace=workspace / "backend",
                    log_path=log_path,
                )
            except ClampBackendError as exc:
                global_error = str(exc)
                chunk_errors = {chunk_id: global_error for chunk_id in chunk_sources}
        else:
            config["resolved_device"] = resolve_device(str(config["device"]))

    for row in chunk_records:
        if row["chunk_id"] in chunk_errors:
            row["error_message"] = chunk_errors[row["chunk_id"]]
    chunk_matrix, aligned_chunks = align_success_rows(
        chunk_records, chunk_vectors, id_field="chunk_id"
    )
    recording_vectors: dict[str, np.ndarray] = {}
    for record in records:
        if record["status"] != "pending":
            continue
        ids = recording_chunks.get(record["recording_id"], [])
        missing = [chunk_id for chunk_id in ids if chunk_id not in chunk_vectors]
        if not ids or missing:
            record["status"] = "failed"
            record["error_message"] = (
                f"{len(missing) or 1} audio chunks failed; recording pooling was not performed"
            )
            continue
        recording_vectors[record["recording_id"]] = np.mean(
            np.stack([chunk_vectors[chunk_id] for chunk_id in ids]), axis=0
        )
    recording_matrix, aligned_records = align_success_rows(
        records, recording_vectors, id_field="recording_id"
    )

    shared_metadata = build_run_metadata(
        config,
        stage="audio",
        checkpoint_path=config["saas_checkpoint_path"],
        model_space="saas",
        run_id=run_id,
        batch_size=int(config["batch_size_audio"]),
        input_manifest_path=config["corpus_manifest"],
        extra={
            "log_path": str(log_path),
            "sample_rate": sample_rate,
            "mono": bool(config.get("audio_mono", True)),
            "chunk_seconds": chunk_seconds,
            "chunk_overlap_seconds": overlap_seconds,
            "aggregation_method": config.get("audio_aggregation", "mean"),
            "authorized_rights_statuses": allowed,
            "copyright_policy": "No audio is downloaded; only existing paths with an explicitly allowed rights_status are read.",
            "global_error": global_error or None,
        },
    )
    chunk_paths = write_embedding_bundle(
        chunks_output,
        chunk_matrix,
        aligned_chunks,
        matrix_filename="chunk_embeddings.npy",
        table_filename="chunk_embeddings.csv",
        fields=CHUNK_FIELDS,
        id_field="chunk_id",
        metadata={**shared_metadata, "stage": "audio_chunks"},
        force=args.force,
        normalize=bool(config["normalize_embeddings"]),
    )
    chunk_store = snapshot_written_bundle(chunk_paths, config)
    paths = write_embedding_bundle(
        output,
        recording_matrix,
        aligned_records,
        matrix_filename="recording_embeddings.npy",
        table_filename="recording_embeddings.csv",
        fields=RECORDING_FIELDS,
        id_field="recording_id",
        metadata={**shared_metadata, "chunk_bundle": "chunks/metadata.json"},
        force=args.force,
        normalize=bool(config["normalize_embeddings"]),
    )
    store = snapshot_written_bundle(paths, config)
    failed = sum(row["status"] == "failed" for row in aligned_records)
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "chunk_paths": {key: str(value) for key, value in chunk_paths.items()},
        "store": {key: str(value) if isinstance(value, Path) else value for key, value in store.items()},
        "chunk_store": {key: str(value) if isinstance(value, Path) else value for key, value in chunk_store.items()},
        "log_path": str(log_path),
        "embedded": recording_matrix.shape[0],
        "chunks_embedded": chunk_matrix.shape[0],
        "failed": failed,
        "unavailable_or_unauthorized": len(records) - recording_matrix.shape[0] - failed,
        "global_error": global_error,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--composer")
    parser.add_argument("--work-id")
    parser.add_argument("--recording-id")
    parser.add_argument("--force", action="store_true")
    add_caffeinate_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        with maybe_caffeinate(args.caffeinate, flags=args.caffeinate_flags):
            result = run(args)
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")
    print(result)
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
