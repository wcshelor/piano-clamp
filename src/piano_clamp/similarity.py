"""Transparent cosine-similarity calculations and output tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


class SimilarityError(ValueError):
    """Raised when embeddings are missing or unsuitable for cosine similarity."""


def as_embedding(vector: np.ndarray, *, label: str = "embedding") -> np.ndarray:
    """Normalize supported global embedding shapes to a finite 1-D vector."""

    array = np.asarray(vector, dtype=np.float64)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise SimilarityError(f"{label} must have shape (D,) or (1, D), got {array.shape}")
    if not np.isfinite(array).all():
        raise SimilarityError(f"{label} contains NaN or infinite values")
    if np.linalg.norm(array) == 0:
        raise SimilarityError(f"{label} has zero norm")
    return array


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return all pairwise cosine similarities between two 2-D arrays."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2:
        raise SimilarityError("cosine inputs must both be 2-D")
    if a.shape[1] != b.shape[1]:
        raise SimilarityError(f"embedding dimensions differ: {a.shape[1]} != {b.shape[1]}")
    a_norms = np.linalg.norm(a, axis=1, keepdims=True)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True)
    if np.any(a_norms == 0) or np.any(b_norms == 0):
        raise SimilarityError("cosine similarity is undefined for zero-norm embeddings")
    return (a / a_norms) @ (b / b_norms).T


def load_named_embeddings(directory: str | Path, ids: Sequence[str]) -> np.ndarray:
    """Load global embeddings named ``<id>.npy`` in a stable order."""

    vectors = []
    root = Path(directory)
    for item_id in ids:
        path = root / f"{item_id}.npy"
        if not path.is_file():
            raise SimilarityError(f"missing embedding: {path}")
        vector = as_embedding(np.load(path, allow_pickle=False), label=str(path))
        if vector.shape[0] != 768:
            raise SimilarityError(f"{path} has dimension {vector.shape[0]}; expected 768")
        vectors.append(vector)
    return np.stack(vectors)


def build_similarity_rows(
    passages: Sequence[Mapping[str, str]],
    prompts: Sequence[Mapping[str, str]],
    values: np.ndarray,
) -> list[dict[str, object]]:
    """Combine an N-by-M matrix with passage and prompt metadata."""

    if values.shape != (len(passages), len(prompts)):
        raise SimilarityError("similarity matrix shape does not match metadata")
    rows: list[dict[str, object]] = []
    for passage_index, passage in enumerate(passages):
        for prompt_index, prompt in enumerate(prompts):
            rows.append(
                {
                    "passage_id": passage["passage_id"],
                    "composer": passage["composer"],
                    "prompt_id": prompt["prompt_id"],
                    "prompt_text": prompt["prompt_text"],
                    "cosine_similarity": float(values[passage_index, prompt_index]),
                }
            )
    return rows


def write_similarity_table(rows: Sequence[Mapping[str, object]], destination: str | Path) -> None:
    """Write the stable prompt-similarity table schema."""

    fields = ("passage_id", "composer", "prompt_id", "prompt_text", "cosine_similarity")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

