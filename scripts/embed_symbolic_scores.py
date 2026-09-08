#!/usr/bin/env python3
"""Embed all supported unique symbolic scores declared by the corpus export."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.clamp_backend import ClampBackend, ClampBackendError, resolve_device  # noqa: E402
from piano_clamp.data_loading import (  # noqa: E402
    filter_manifest_rows,
    load_pipeline_config,
    read_corpus_manifest,
    select_symbolic_records,
)
from piano_clamp.embedding_io import (  # noqa: E402
    align_success_rows,
    guard_bundle,
    sha256_path,
    write_embedding_bundle,
)
from piano_clamp.embedding_store import snapshot_written_bundle  # noqa: E402
from piano_clamp.metadata import build_run_metadata, make_run_id, stage_log_path  # noqa: E402
from piano_clamp.power import add_caffeinate_arguments, maybe_caffeinate  # noqa: E402


TABLE_FIELDS = (
    "score_id",
    "composer",
    "period",
    "composition_id",
    "work_id",
    "movement_id",
    "title",
    "source_path",
    "source_format",
    "content_hash",
    "embedding_row",
    "embedding_dimension",
    "status",
    "error_message",
)


def run(args: argparse.Namespace) -> dict[str, object]:
    config = load_pipeline_config(args.config, device=args.device)
    run_id = make_run_id("symbolic", config)
    rows = filter_manifest_rows(
        read_corpus_manifest(config["corpus_manifest"]),
        composer=args.composer,
        work_id=args.work_id,
    )
    records = select_symbolic_records(rows, config["corpus_root"])
    if args.limit is not None:
        records = records[: args.limit]
    output = Path(config["output_root"]) / "symbolic"
    guard_bundle(
        output,
        ("score_embeddings.npy", "score_embeddings.csv", "metadata.json"),
        force=args.force,
    )

    staged_by_hash: dict[str, tuple[str, Path]] = {}
    aliases: dict[str, str] = {}
    for record in records:
        source = record.pop("resolved_source_path", None)
        record["content_hash"] = ""
        if record["status"] != "pending" or not source:
            continue
        digest = sha256_path(source)
        record["content_hash"] = digest
        if digest in staged_by_hash:
            aliases[record["score_id"]] = staged_by_hash[digest][0]
        else:
            staged_by_hash[digest] = (record["score_id"], Path(source))

    vectors = {}
    errors: dict[str, str] = {}
    global_error = ""
    if staged_by_hash:
        try:
            backend = ClampBackend(config, model_space="c2")
            config["resolved_device"] = backend.device
            temporary_root = Path(config["temporary_root"])
            temporary_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="symbolic-", dir=temporary_root) as temporary:
                vectors, errors = backend.embed_symbolic(
                    {item_id: source for item_id, source in staged_by_hash.values()},
                    workspace=Path(temporary),
                    log_path=stage_log_path(config, "embed_symbolic", run_id=run_id),
                )
        except ClampBackendError as exc:
            global_error = str(exc)
            errors.update({item_id: global_error for item_id, _ in staged_by_hash.values()})
    else:
        config["resolved_device"] = resolve_device(str(config["device"]))
    for alias, canonical in aliases.items():
        if canonical in vectors:
            vectors[alias] = vectors[canonical]
        elif canonical in errors:
            errors[alias] = errors[canonical]
    for record in records:
        item_error = errors.get(record["score_id"])
        if item_error:
            record["error_message"] = item_error
    matrix, aligned = align_success_rows(records, vectors, id_field="score_id")
    metadata = build_run_metadata(
        config,
        stage="symbolic",
        checkpoint_path=config["checkpoint_path"],
        model_space="c2",
        run_id=run_id,
        batch_size=int(config["batch_size_symbolic"]),
        input_manifest_path=config["corpus_manifest"],
        extra={
            "log_path": str(stage_log_path(config, "embed_symbolic", run_id=run_id)),
            "filter": {"composer": args.composer, "work_id": args.work_id, "limit": args.limit},
            "supported_source_formats": ["mxl", "musicxml", "xml", "mid", "midi"],
            "deduplicated_content_count": len(aliases),
            "global_error": global_error or None,
        },
    )
    paths = write_embedding_bundle(
        output,
        matrix,
        aligned,
        matrix_filename="score_embeddings.npy",
        table_filename="score_embeddings.csv",
        fields=TABLE_FIELDS,
        id_field="score_id",
        metadata=metadata,
        force=args.force,
        normalize=bool(config["normalize_embeddings"]),
    )
    store = snapshot_written_bundle(paths, config)
    failed = sum(record["status"] == "failed" for record in aligned)
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "store": {key: str(value) if isinstance(value, Path) else value for key, value in store.items()},
        "log_path": str(stage_log_path(config, "embed_symbolic", run_id=run_id)),
        "embedded": matrix.shape[0],
        "failed": failed,
        "unsupported_or_unavailable": len(records) - matrix.shape[0] - failed,
        "global_error": global_error,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--composer")
    parser.add_argument("--work-id")
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
