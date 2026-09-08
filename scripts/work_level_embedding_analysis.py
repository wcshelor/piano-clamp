#!/usr/bin/env python3
"""Work-level composer geometry and leave-one-work-out embedding probes."""

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


def _describe(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"count": 0, "mean": None, "std": None, "median": None, "q05": None, "q25": None, "q75": None, "q95": None}
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
    }


def _read_passages(path: str) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    matrix, rows, metadata = read_embedding_bundle(path)
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["embedding_row"].astype(str) != ""].copy()
    frame = frame.sort_values("embedding_row", key=lambda col: col.astype(int)).reset_index(drop=True)
    return _normalize(matrix), frame, metadata


def _work_table(passages: pd.DataFrame, matrix: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    required = ["composer", "work_id", "movement_id", "recording_id", "passage_generation_method", "audio_origin"]
    for column in required:
        if column not in passages.columns:
            passages[column] = ""
    for work_id, group in passages.groupby("work_id", sort=True):
        indices = group.index.to_numpy()
        vector = _normalize(matrix[indices].mean(axis=0, keepdims=True))[0]
        vectors.append(vector)
        records.append(
            {
                "work_id": work_id,
                "composer": group["composer"].mode().iloc[0],
                "passage_count": int(len(group)),
                "movement_count": int(group["movement_id"].nunique()),
                "recording_count": int(group["recording_id"].nunique()),
                "audio_origin_count": int(group["audio_origin"].nunique()),
                "passage_methods": ";".join(sorted(group["passage_generation_method"].dropna().unique())),
            }
        )
    return pd.DataFrame(records), np.stack(vectors).astype(np.float32)


def _pairwise_geometry(table: pd.DataFrame, matrix: np.ndarray) -> dict[str, Any]:
    sims = matrix @ matrix.T
    left, right = np.triu_indices(matrix.shape[0], k=1)
    composers = table["composer"].to_numpy(str)
    values = sims[left, right]
    within = composers[left] == composers[right]
    result = {
        "all_work_pairs": _describe(values),
        "within_composer_work_pairs": _describe(values[within]),
        "between_composer_work_pairs": _describe(values[~within]),
    }
    for composer in sorted(set(composers)):
        mask = (composers[left] == composer) & (composers[right] == composer)
        result[f"within_{composer}_work_pairs"] = _describe(values[mask])
    return result


def _leave_one_work_out(table: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    composers = table["composer"].to_numpy(str)
    work_ids = table["work_id"].to_numpy(str)
    rows = []
    for test_index in range(matrix.shape[0]):
        train = np.ones(matrix.shape[0], dtype=bool)
        train[test_index] = False
        centroids = {}
        for composer in sorted(set(composers[train])):
            member = train & (composers == composer)
            centroids[composer] = _normalize(matrix[member].mean(axis=0, keepdims=True))[0]
        scores = {composer: float(matrix[test_index] @ centroid) for composer, centroid in centroids.items()}
        predicted = max(scores, key=scores.get)
        rows.append(
            {
                "work_id": work_ids[test_index],
                "composer": composers[test_index],
                "predicted_composer": predicted,
                "correct": bool(predicted == composers[test_index]),
                "similarity_to_chopin_centroid": scores.get(CHOPIN, np.nan),
                "similarity_to_mozart_centroid": scores.get(MOZART, np.nan),
                "chopin_minus_mozart_score": scores.get(CHOPIN, np.nan) - scores.get(MOZART, np.nan),
            }
        )
    return pd.DataFrame(rows)


def _passage_method_work_sensitivity(passages: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    rows = []
    for method, group in passages.groupby("passage_generation_method", sort=True):
        if group["composer"].nunique() < 2 or group["work_id"].nunique() < 3:
            continue
        work_rows, work_matrix = _work_table(group.reset_index(drop=True), matrix[group.index.to_numpy()])
        loo = _leave_one_work_out(work_rows, work_matrix)
        geometry = _pairwise_geometry(work_rows, work_matrix)
        rows.append(
            {
                "passage_generation_method": method,
                "work_count": int(len(work_rows)),
                "passage_count": int(len(group)),
                "loo_accuracy": float(loo["correct"].mean()),
                "within_mean": geometry["within_composer_work_pairs"]["mean"],
                "between_mean": geometry["between_composer_work_pairs"]["mean"],
                "within_minus_between": (
                    geometry["within_composer_work_pairs"]["mean"]
                    - geometry["between_composer_work_pairs"]["mean"]
                ),
            }
        )
    return pd.DataFrame(rows)


def _prompt_work_axis(
    prompt_bundle: str | None,
    work_table: pd.DataFrame,
    work_matrix: np.ndarray,
) -> pd.DataFrame:
    if not prompt_bundle:
        return pd.DataFrame()
    prompt_matrix, prompt_rows, _ = read_embedding_bundle(prompt_bundle)
    prompts = pd.DataFrame(prompt_rows)
    prompts = prompts.loc[prompts["embedding_row"].astype(str) != ""].copy()
    prompts = prompts.sort_values("embedding_row", key=lambda col: col.astype(int)).reset_index(drop=True)
    prompt_matrix = _normalize(prompt_matrix)
    centroids = {
        composer: _normalize(work_matrix[work_table["composer"].to_numpy(str) == composer].mean(axis=0, keepdims=True))[0]
        for composer in sorted(work_table["composer"].unique())
    }
    rows = []
    for index, prompt in prompts.iterrows():
        row = {
            "prompt_id": prompt["prompt_id"],
            "family": prompt.get("family", ""),
            "subfamily": prompt.get("subfamily", ""),
            "prompt_text": prompt.get("prompt_text", ""),
        }
        for composer, centroid in centroids.items():
            row[f"similarity_to_{composer}"] = float(prompt_matrix[index] @ centroid)
        if CHOPIN in centroids and MOZART in centroids:
            row["chopin_minus_mozart"] = row[f"similarity_to_{CHOPIN}"] - row[f"similarity_to_{MOZART}"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("chopin_minus_mozart", ascending=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    tables = output / "tables"
    reports = output / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    passage_matrix, passages, metadata = _read_passages(args.passage_bundle)
    work_table, work_matrix = _work_table(passages, passage_matrix)
    work_table.to_csv(tables / "work_embedding_manifest.csv", index=False)
    np.save(tables / "work_embeddings.npy", work_matrix, allow_pickle=False)

    loo = _leave_one_work_out(work_table, work_matrix)
    loo.to_csv(tables / "leave_one_work_out_nearest_centroid.csv", index=False)

    method_sensitivity = _passage_method_work_sensitivity(passages, passage_matrix)
    method_sensitivity.to_csv(tables / "passage_method_work_sensitivity.csv", index=False)

    prompt_axis = _prompt_work_axis(args.prompt_bundle, work_table, work_matrix)
    if not prompt_axis.empty:
        prompt_axis.to_csv(tables / "prompt_work_centroid_axis_projection.csv", index=False)

    geometry = _pairwise_geometry(work_table, work_matrix)
    report = {
        "passage_bundle": str(Path(args.passage_bundle).resolve()),
        "prompt_bundle": str(Path(args.prompt_bundle).resolve()) if args.prompt_bundle else None,
        "output_dir": str(output.resolve()),
        "passage_metadata": {
            "row_count": metadata.get("row_count"),
            "model_space": metadata.get("model_space"),
            "run_id": metadata.get("run_id"),
        },
        "work_count": int(len(work_table)),
        "composer_work_counts": work_table["composer"].value_counts().to_dict(),
        "composer_passage_counts": passages["composer"].value_counts().to_dict(),
        "leave_one_work_out": {
            "accuracy": float(loo["correct"].mean()),
            "correct": int(loo["correct"].sum()),
            "total": int(len(loo)),
            "confusion": (
                loo.groupby(["composer", "predicted_composer"]).size().rename("count").reset_index().to_dict(orient="records")
            ),
        },
        "work_level_geometry": geometry,
    }
    if not method_sensitivity.empty:
        report["passage_method_sensitivity"] = method_sensitivity.to_dict(orient="records")
    (reports / "work_level_embedding_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passage-bundle", required=True)
    parser.add_argument("--prompt-bundle")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run(args)
    print(
        json.dumps(
            {
                "output_dir": report["output_dir"],
                "work_count": report["work_count"],
                "leave_one_work_out": report["leave_one_work_out"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
