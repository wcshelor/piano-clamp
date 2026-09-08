"""Atomic, aligned storage for embedding matrices and tidy metadata tables."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


class EmbeddingIOError(RuntimeError):
    """Raised when an embedding bundle is incomplete or would be overwritten."""


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2:
        raise EmbeddingIOError(f"embedding matrix must be 2-D, got {array.shape}")
    if not np.isfinite(array).all():
        raise EmbeddingIOError("embedding matrix contains NaN or infinite values")
    if not array.size:
        return array
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise EmbeddingIOError("cannot normalize a zero embedding")
    return array / norms


def sha256_path(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def guard_bundle(
    directory: str | Path,
    filenames: Sequence[str],
    *,
    force: bool,
) -> None:
    root = Path(directory)
    existing = [root / name for name in filenames if (root / name).exists()]
    if existing and not force:
        rendered = ", ".join(path.name for path in existing)
        raise EmbeddingIOError(
            f"refusing to overwrite existing outputs ({rendered}); pass --force explicitly"
        )


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: "" if row.get(field) is None else str(row.get(field, ""))
                        for field in fields
                    }
                )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def align_success_rows(
    rows: Sequence[Mapping[str, Any]],
    vectors: Mapping[str, np.ndarray],
    *,
    id_field: str,
    expected_dimension: int = 768,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Assign stable matrix rows to successes while retaining explicit failures."""

    output_rows: list[dict[str, Any]] = []
    ordered_vectors: list[np.ndarray] = []
    for source in rows:
        row = dict(source)
        item_id = str(row.get(id_field, ""))
        vector = vectors.get(item_id)
        if vector is None:
            row["embedding_row"] = ""
            row["embedding_dimension"] = ""
            if row.get("status") in {None, "", "pending"}:
                row["status"] = "failed"
                row.setdefault("error_message", "embedding was not produced")
        else:
            array = np.asarray(vector, dtype=np.float32)
            if array.ndim == 2 and array.shape[0] == 1:
                array = array[0]
            if array.shape != (expected_dimension,):
                raise EmbeddingIOError(
                    f"{item_id} has shape {array.shape}; expected ({expected_dimension},)"
                )
            if not np.isfinite(array).all():
                raise EmbeddingIOError(f"{item_id} contains NaN or infinite values")
            row["embedding_row"] = len(ordered_vectors)
            row["embedding_dimension"] = expected_dimension
            row["status"] = "success"
            row["error_message"] = ""
            ordered_vectors.append(array)
        output_rows.append(row)
    matrix = (
        np.stack(ordered_vectors).astype(np.float32, copy=False)
        if ordered_vectors
        else np.empty((0, expected_dimension), dtype=np.float32)
    )
    return matrix, output_rows


def write_embedding_bundle(
    directory: str | Path,
    matrix: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    *,
    matrix_filename: str,
    table_filename: str,
    metadata_filename: str = "metadata.json",
    fields: Sequence[str],
    id_field: str,
    metadata: Mapping[str, Any],
    force: bool = False,
    normalize: bool = True,
) -> dict[str, Path]:
    root = Path(directory)
    names = (matrix_filename, table_filename, metadata_filename)
    guard_bundle(root, names, force=force)
    array = normalize_rows(matrix) if normalize else np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise EmbeddingIOError("embedding matrix must be finite and two-dimensional")

    successful = [row for row in rows if str(row.get("embedding_row", "")) != ""]
    if len(successful) != array.shape[0]:
        raise EmbeddingIOError(
            f"metadata success rows ({len(successful)}) do not match matrix rows ({array.shape[0]})"
        )
    indices = [int(row["embedding_row"]) for row in successful]
    if indices != list(range(array.shape[0])):
        raise EmbeddingIOError("embedding_row values are not contiguous and ordered")

    matrix_path = root / matrix_filename
    table_path = root / table_filename
    metadata_path = root / metadata_filename
    _atomic_npy(matrix_path, array)
    _atomic_csv(table_path, rows, fields)
    document = dict(metadata)
    document.update(
        {
            "schema_version": 1,
            "matrix_file": matrix_filename,
            "table_file": table_filename,
            "id_field": id_field,
            "embedding_shape": list(array.shape),
            "embedding_dimension": int(array.shape[1]),
            "row_count": int(array.shape[0]),
            "metadata_record_count": len(rows),
            "failure_count": len(rows) - len(successful),
            "normalization": "l2" if normalize else "none",
            "overwrite_guard": "existing bundles require explicit force=True",
            "embedding_sha256": sha256_path(matrix_path),
        }
    )
    _atomic_json(metadata_path, document)
    return {"matrix": matrix_path, "table": table_path, "metadata": metadata_path}


def read_embedding_bundle(directory: str | Path) -> tuple[np.ndarray, list[dict[str, str]], dict[str, Any]]:
    root = Path(directory)
    metadata_path = root / "metadata.json"
    if not metadata_path.is_file():
        raise EmbeddingIOError(f"embedding metadata does not exist: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EmbeddingIOError(f"invalid embedding metadata: {metadata_path}") from exc
    matrix_path = root / str(metadata.get("matrix_file", ""))
    table_path = root / str(metadata.get("table_file", ""))
    if not matrix_path.is_file() or not table_path.is_file():
        raise EmbeddingIOError(f"embedding bundle is incomplete: {root}")
    matrix = np.load(matrix_path, allow_pickle=False)
    with table_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return matrix, rows, metadata
