#!/usr/bin/env python3
"""Rank Chopin Ballade passage windows by proximity to the Mozart centroid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.embedding_io import read_embedding_bundle  # noqa: E402


CHOPIN = "Frédéric Chopin"
MOZART = "Wolfgang Amadeus Mozart"


def _normalize(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize zero-length embedding vectors")
    return array / norms


def _load_bundle(path: str | Path) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    matrix, rows, metadata = read_embedding_bundle(path)
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["embedding_row"].astype(str) != ""].copy()
    frame = frame.sort_values("embedding_row", key=lambda col: col.astype(int)).reset_index(drop=True)
    return _normalize(matrix), frame, metadata


def _work_vectors(passages: pd.DataFrame, matrix: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for work_id, group in passages.groupby("work_id", sort=True):
        indices = group.index.to_numpy()
        vector = _normalize(matrix[indices].mean(axis=0, keepdims=True))[0]
        vectors.append(vector)
        records.append(
            {
                "work_id": work_id,
                "work_title": group["work_title"].mode().iloc[0],
                "composer": group["composer"].mode().iloc[0],
                "passage_count": int(len(group)),
                "recording_count": int(group["recording_id"].nunique()) if "recording_id" in group else 0,
            }
        )
    return pd.DataFrame(records), np.stack(vectors).astype(np.float32)


def _add_work_titles(passages: pd.DataFrame, manifest_path: str | Path) -> pd.DataFrame:
    if "work_title" in passages.columns and passages["work_title"].notna().any():
        return passages
    manifest = (
        pd.read_csv(manifest_path, usecols=["work_id", "work_title"])
        .dropna(subset=["work_id"])
        .drop_duplicates("work_id")
    )
    return passages.merge(manifest, on="work_id", how="left")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    tables = output / "tables"
    reports = output / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    matrix, passages, metadata = _load_bundle(args.passage_bundle)
    passages = _add_work_titles(passages, args.manifest)
    for column in (
        "passage_id",
        "composer",
        "work_id",
        "work_title",
        "recording_id",
        "audio_origin",
        "passage_generation_method",
        "bar_start",
        "bar_end",
    ):
        if column not in passages.columns:
            passages[column] = ""

    works, work_matrix = _work_vectors(passages, matrix)
    composers = works["composer"].to_numpy(str)
    if not np.any(composers == CHOPIN) or not np.any(composers == MOZART):
        raise ValueError("both Chopin and Mozart work vectors are required")

    chopin_centroid = _normalize(work_matrix[composers == CHOPIN].mean(axis=0, keepdims=True))[0]
    mozart_centroid = _normalize(work_matrix[composers == MOZART].mean(axis=0, keepdims=True))[0]

    target = passages[
        passages["composer"].eq(CHOPIN)
        & passages["work_title"].astype(str).str.contains(args.title_pattern, case=False, na=False)
    ].copy()
    if target.empty:
        raise ValueError(f"no Chopin passages matched title pattern: {args.title_pattern}")
    target_indices = target.index.to_numpy()
    target["similarity_to_mozart_centroid"] = matrix[target_indices] @ mozart_centroid
    target["similarity_to_chopin_centroid"] = matrix[target_indices] @ chopin_centroid
    target["mozart_minus_chopin"] = (
        target["similarity_to_mozart_centroid"] - target["similarity_to_chopin_centroid"]
    )

    columns = [
        "work_title",
        "work_id",
        "passage_id",
        "recording_id",
        "audio_origin",
        "passage_generation_method",
        "bar_start",
        "bar_end",
        "similarity_to_mozart_centroid",
        "similarity_to_chopin_centroid",
        "mozart_minus_chopin",
    ]
    absolute = target[columns].sort_values("similarity_to_mozart_centroid", ascending=False)
    relative = target[columns].sort_values("mozart_minus_chopin", ascending=False)
    absolute.to_csv(tables / "ballade_windows_ranked_by_mozart_centroid.csv", index=False)
    relative.to_csv(tables / "ballade_windows_ranked_by_relative_mozartness.csv", index=False)

    summary = (
        target.groupby(["work_title", "passage_generation_method"], as_index=False)
        .agg(
            windows=("passage_id", "count"),
            mean_mozart_similarity=("similarity_to_mozart_centroid", "mean"),
            max_mozart_similarity=("similarity_to_mozart_centroid", "max"),
            mean_relative_mozartness=("mozart_minus_chopin", "mean"),
            max_relative_mozartness=("mozart_minus_chopin", "max"),
        )
        .sort_values(["work_title", "passage_generation_method"])
    )
    summary.to_csv(tables / "ballade_window_length_summary.csv", index=False)

    by_work = (
        target.groupby("work_title", as_index=False)
        .agg(
            windows=("passage_id", "count"),
            mean_mozart_similarity=("similarity_to_mozart_centroid", "mean"),
            max_mozart_similarity=("similarity_to_mozart_centroid", "max"),
            mean_relative_mozartness=("mozart_minus_chopin", "mean"),
            max_relative_mozartness=("mozart_minus_chopin", "max"),
        )
        .sort_values("mean_mozart_similarity", ascending=False)
    )
    by_work.to_csv(tables / "ballade_work_mozartness_summary.csv", index=False)

    report = {
        "passage_bundle": str(Path(args.passage_bundle).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "output_dir": str(output.resolve()),
        "title_pattern": args.title_pattern,
        "bundle_metadata": {
            "row_count": metadata.get("row_count"),
            "model_space": metadata.get("model_space"),
            "run_id": metadata.get("run_id"),
        },
        "matched_windows": int(len(target)),
        "matched_works": int(target["work_id"].nunique()),
        "top_absolute_mozart_windows": absolute.head(args.report_top_n).to_dict(orient="records"),
        "top_relative_mozart_windows": relative.head(args.report_top_n).to_dict(orient="records"),
        "work_summary": by_work.to_dict(orient="records"),
    }
    (reports / "ballade_window_mozartness_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passage-bundle", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title-pattern", default="Ballades")
    parser.add_argument("--report-top-n", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run(args)
    print(
        json.dumps(
            {
                "output_dir": report["output_dir"],
                "matched_windows": report["matched_windows"],
                "matched_works": report["matched_works"],
                "top_absolute_mozart_window": report["top_absolute_mozart_windows"][:1],
                "top_relative_mozart_window": report["top_relative_mozart_windows"][:1],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
