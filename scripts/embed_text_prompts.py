#!/usr/bin/env python3
"""Embed the enabled canonical prompt bank with the configured CLaMP checkpoint."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.clamp_backend import ClampBackend, ClampBackendError  # noqa: E402
from piano_clamp.data_loading import load_pipeline_config, load_prompt_bank  # noqa: E402
from piano_clamp.embedding_io import (  # noqa: E402
    align_success_rows,
    content_hash,
    guard_bundle,
    write_embedding_bundle,
)
from piano_clamp.embedding_store import snapshot_written_bundle  # noqa: E402
from piano_clamp.metadata import build_run_metadata, make_run_id, stage_log_path  # noqa: E402
from piano_clamp.power import add_caffeinate_arguments, maybe_caffeinate  # noqa: E402


TABLE_FIELDS = (
    "prompt_id",
    "family",
    "subfamily",
    "prompt_text",
    "embedding_row",
    "embedding_dimension",
    "polarity",
    "composer_reference",
    "interpretive_level",
    "enabled",
    "notes",
    "content_hash",
    "status",
    "error_message",
)


def run(args: argparse.Namespace) -> dict[str, object]:
    config = load_pipeline_config(args.config, device=args.device)
    run_id = make_run_id("text", config)
    rows = load_prompt_bank(config["prompt_bank"], enabled_only=True)
    if args.limit is not None:
        rows = rows[: args.limit]
    model_space = args.model_space
    output = Path(config["output_root"]) / ("text" if model_space == "c2" else "text_saas")
    guard_bundle(
        output,
        ("prompt_embeddings.npy", "prompt_embeddings.csv", "metadata.json"),
        force=args.force,
    )
    records = []
    for row in rows:
        record = dict(row)
        record.update(
            {
                "content_hash": content_hash([row["prompt_id"], row["prompt_text"]]),
                "status": "pending",
                "error_message": "",
            }
        )
        records.append(record)

    vectors = {}
    errors: dict[str, str] = {}
    backend = ClampBackend(config, model_space=model_space)
    config["resolved_device"] = backend.device
    temporary_root = Path(config["temporary_root"])
    temporary_root.mkdir(parents=True, exist_ok=True)
    log_path = stage_log_path(config, f"embed_text_{model_space}", run_id=run_id)
    with tempfile.TemporaryDirectory(prefix="text-", dir=temporary_root) as temporary:
        vectors, errors = backend.embed_texts(
            {row["prompt_id"]: row["prompt_text"] for row in records},
            workspace=Path(temporary),
            log_path=log_path,
        )
    for record in records:
        if record["prompt_id"] in errors:
            record["error_message"] = errors[record["prompt_id"]]
    matrix, aligned = align_success_rows(records, vectors, id_field="prompt_id")
    checkpoint_key = "checkpoint_path" if model_space == "c2" else "saas_checkpoint_path"
    metadata = build_run_metadata(
        config,
        stage="text",
        checkpoint_path=config[checkpoint_key],
        model_space=model_space,
        run_id=run_id,
        batch_size=int(config["batch_size_text"]),
        prompt_bank_path=config["prompt_bank"],
        extra={
            "log_path": str(log_path),
            "prompt_records": rows,
            "prompt_order_preserved": True,
            "upstream_batching_note": "All prompts are staged together and processed in one model process; upstream inference is item-serial.",
        },
    )
    paths = write_embedding_bundle(
        output,
        matrix,
        aligned,
        matrix_filename="prompt_embeddings.npy",
        table_filename="prompt_embeddings.csv",
        fields=TABLE_FIELDS,
        id_field="prompt_id",
        metadata=metadata,
        force=args.force,
        normalize=bool(config["normalize_embeddings"]),
    )
    store = snapshot_written_bundle(paths, config)
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "store": {key: str(value) if isinstance(value, Path) else value for key, value in store.items()},
        "log_path": str(log_path),
        "embedded": matrix.shape[0],
        "failed": len(records) - matrix.shape[0],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--model-space", choices=("c2", "saas"), default="c2")
    parser.add_argument("--limit", type=int)
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
