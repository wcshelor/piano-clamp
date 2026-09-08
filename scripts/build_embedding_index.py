#!/usr/bin/env python3
"""Build one deterministic registry of all embedding rows and explicit failures."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.data_loading import load_pipeline_config  # noqa: E402
from piano_clamp.embedding_io import EmbeddingIOError, read_embedding_bundle  # noqa: E402


FIELDS = (
    "item_id",
    "item_type",
    "model_space",
    "bundle_path",
    "embedding_row",
    "embedding_dimension",
    "status",
    "composer",
    "composition_id",
    "work_id",
    "movement_id",
    "recording_id",
    "source_path",
    "content_hash",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_pipeline_config(args.config)
        root = Path(config["output_root"])
        output = root / "embedding_index.csv"
        metadata_output = root / "embedding_index.json"
        if (output.exists() or metadata_output.exists()) and not args.force:
            raise RuntimeError("embedding index already exists; pass --force explicitly")
        locations = (
            ("text", root / "text"),
            ("text_saas", root / "text_saas"),
            ("score", root / "symbolic"),
            ("audio", root / "audio"),
            ("passage_score", root / "passages"),
            ("passage_performance_midi", root / "passages" / "performance_midi"),
            ("audio_passage", root / "passages" / "audio"),
        )
        index_rows = []
        skipped = []
        for item_type, directory in locations:
            try:
                _, rows, metadata = read_embedding_bundle(directory)
            except EmbeddingIOError as exc:
                skipped.append(str(exc))
                continue
            id_field = metadata["id_field"]
            for row in rows:
                material = row.get("source_material") or row.get("embedding_modality", "")
                source_path = row.get("source_path", "")
                if not source_path:
                    if material == "performance_midi":
                        source_path = row.get("performance_midi_path", "")
                    elif material == "audio":
                        source_path = row.get("audio_path", "")
                    else:
                        source_path = row.get("score_path") or row.get("midi_path", "")
                index_rows.append(
                    {
                        "item_id": row.get(id_field, ""),
                        "item_type": item_type,
                        "model_space": metadata.get("model_space", ""),
                        "bundle_path": directory.relative_to(root).as_posix() or ".",
                        "embedding_row": row.get("embedding_row", ""),
                        "embedding_dimension": row.get("embedding_dimension", ""),
                        "status": row.get("status", ""),
                        "composer": row.get("composer", ""),
                        "composition_id": row.get("composition_id", ""),
                        "work_id": row.get("work_id", ""),
                        "movement_id": row.get("movement_id", ""),
                        "recording_id": row.get("recording_id", ""),
                        "source_path": source_path,
                        "content_hash": row.get("content_hash", ""),
                    }
                )
        index_rows.sort(key=lambda row: (row["model_space"], row["item_type"], row["item_id"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(index_rows)
        metadata_output.write_text(
            json.dumps(
                {
                    "row_count": len(index_rows),
                    "successful_count": sum(bool(row["embedding_row"] != "") for row in index_rows),
                    "skipped_bundles": skipped,
                    "ordering": ["model_space", "item_type", "item_id"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
