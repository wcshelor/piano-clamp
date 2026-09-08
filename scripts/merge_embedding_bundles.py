#!/usr/bin/env python3
"""Combine compatible embedding bundles without modifying their source bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from piano_clamp.embedding_io import EmbeddingIOError, read_embedding_bundle, write_embedding_bundle  # noqa: E402


def _fields(rows: list[dict[str, str]]) -> list[str]:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def merge(inputs: list[Path], output: Path) -> dict[str, Any]:
    matrices: list[np.ndarray] = []
    rows_out: list[dict[str, str]] = []
    source_metadata: list[dict[str, Any]] = []
    model_space: str | None = None
    id_field: str | None = None

    for source in inputs:
        matrix, rows, metadata = read_embedding_bundle(source)
        source_space = str(metadata.get("model_space", ""))
        source_id_field = str(metadata.get("id_field", ""))
        if not source_space or not source_id_field:
            raise EmbeddingIOError(f"{source} has incomplete embedding metadata")
        if model_space is not None and source_space != model_space:
            raise EmbeddingIOError("all input bundles must share one model_space")
        if id_field is not None and source_id_field != id_field:
            raise EmbeddingIOError("all input bundles must share one id_field")
        model_space, id_field = source_space, source_id_field

        successful = sorted(
            (dict(row) for row in rows if row.get("embedding_row", "") != ""),
            key=lambda row: int(row["embedding_row"]),
        )
        if len(successful) != matrix.shape[0]:
            raise EmbeddingIOError(f"{source} table/matrix rows are not aligned")
        offset = sum(item.shape[0] for item in matrices)
        for index, row in enumerate(successful):
            row["embedding_row"] = str(offset + index)
            rows_out.append(row)
        matrices.append(np.asarray(matrix, dtype=np.float32))
        source_metadata.append({"path": str(source), "rows": len(successful), "metadata": metadata})

    if not matrices or model_space is None or id_field is None:
        raise EmbeddingIOError("at least one input bundle is required")
    combined = np.concatenate(matrices, axis=0)
    metadata = {
        "model_space": model_space,
        "id_field": id_field,
        "source_material": "performance_midi",
        "merged_sources": source_metadata,
        "merged_embedding_count": int(combined.shape[0]),
    }
    write_embedding_bundle(
        output,
        combined,
        rows_out,
        matrix_filename="passage_embeddings.npy",
        table_filename="passage_embeddings.csv",
        fields=_fields(rows_out),
        id_field=id_field,
        metadata=metadata,
        normalize=False,
    )
    return {"output": str(output), "rows": len(rows_out), "shape": list(combined.shape)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(merge(args.input, args.output))
    except (EmbeddingIOError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
