#!/usr/bin/env python3
"""Advanced prompt clustering, cross-fitted axes, and source-origin checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

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


def _load_bundle(path: str) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    matrix, rows, metadata = read_embedding_bundle(path)
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["embedding_row"].astype(str) != ""].copy()
    frame = frame.sort_values("embedding_row", key=lambda col: col.astype(int)).reset_index(drop=True)
    return _normalize(matrix), frame, metadata


def _work_table(passages: pd.DataFrame, matrix: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    vectors = []
    for column in ("composer", "work_id", "recording_id", "audio_origin", "passage_generation_method"):
        if column not in passages.columns:
            passages[column] = ""
    for work_id, group in passages.groupby("work_id", sort=True):
        indices = group.index.to_numpy()
        vectors.append(_normalize(matrix[indices].mean(axis=0, keepdims=True))[0])
        origins = sorted(group["audio_origin"].dropna().astype(str).unique())
        rows.append(
            {
                "work_id": work_id,
                "composer": group["composer"].mode().iloc[0],
                "passage_count": int(len(group)),
                "recording_count": int(group["recording_id"].nunique()),
                "audio_origin_count": int(len(origins)),
                "audio_origins": ";".join(origins),
            }
        )
    return pd.DataFrame(rows), np.stack(vectors).astype(np.float32)


def _prompt_response_profiles(
    prompt_matrix: np.ndarray,
    prompts: pd.DataFrame,
    passage_matrix: np.ndarray,
    passages: pd.DataFrame,
) -> pd.DataFrame:
    values = prompt_matrix @ passage_matrix.T
    rows = []
    for prompt_index, prompt in prompts.iterrows():
        row: dict[str, Any] = {
            "prompt_id": prompt["prompt_id"],
            "family": prompt.get("family", ""),
            "subfamily": prompt.get("subfamily", ""),
            "prompt_text": prompt.get("prompt_text", ""),
        }
        for composer in sorted(passages["composer"].dropna().unique()):
            mask = passages["composer"].to_numpy(str) == composer
            row[f"mean_similarity_{composer}"] = float(values[prompt_index, mask].mean())
        if CHOPIN in passages["composer"].values and MOZART in passages["composer"].values:
            row["passage_response_chopin_minus_mozart"] = (
                row[f"mean_similarity_{CHOPIN}"] - row[f"mean_similarity_{MOZART}"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _cluster_labels(features: np.ndarray, k: int) -> np.ndarray:
    try:
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=k, affinity="cosine", linkage="average")
    return model.fit_predict(features)


def _prompt_clustering(
    prompt_matrix: np.ndarray,
    prompts: pd.DataFrame,
    response_profiles: pd.DataFrame,
    output_tables: Path,
    *,
    min_k: int,
    max_k: int,
) -> dict[str, Any]:
    response_cols = [col for col in response_profiles.columns if col.startswith("mean_similarity_") or col.endswith("_minus_mozart")]
    response_features = _normalize(response_profiles[response_cols].to_numpy(float))
    rows = []
    assignments: dict[str, pd.DataFrame] = {}
    for feature_name, features in (("embedding", prompt_matrix), ("response_profile", response_features)):
        upper = min(max_k, len(prompts) - 1)
        for k in range(min_k, upper + 1):
            labels = _cluster_labels(features, k)
            score = silhouette_score(features, labels, metric="cosine") if len(set(labels)) > 1 else np.nan
            rows.append({"feature_space": feature_name, "k": k, "silhouette_cosine": float(score)})
            frame = prompts[["prompt_id", "family", "subfamily", "prompt_text"]].copy()
            frame["cluster"] = labels
            assignments[f"{feature_name}_k{k}"] = frame
    scores = pd.DataFrame(rows)
    scores.to_csv(output_tables / "prompt_cluster_model_scores.csv", index=False)
    best = scores.sort_values(["silhouette_cosine", "k"], ascending=[False, True]).groupby("feature_space").head(1)
    for _, row in best.iterrows():
        key = f"{row['feature_space']}_k{int(row['k'])}"
        assignments[key].to_csv(output_tables / f"prompt_clusters_{key}.csv", index=False)
    embedding_labels = assignments[f"embedding_k{int(best.loc[best['feature_space'] == 'embedding', 'k'].iloc[0])}"]["cluster"]
    response_labels = assignments[f"response_profile_k{int(best.loc[best['feature_space'] == 'response_profile', 'k'].iloc[0])}"]["cluster"]
    return {
        "best_models": best.to_dict(orient="records"),
        "embedding_response_cluster_adjusted_rand": float(adjusted_rand_score(embedding_labels, response_labels)),
    }


def _cross_fitted_axis(
    work_table: pd.DataFrame,
    work_matrix: np.ndarray,
    prompt_matrix: np.ndarray,
    prompts: pd.DataFrame,
    passages: pd.DataFrame,
    passage_matrix: np.ndarray,
    output_tables: Path,
) -> dict[str, Any]:
    composers = work_table["composer"].to_numpy(str)
    work_ids = work_table["work_id"].to_numpy(str)
    prompt_rows = []
    work_rows = []
    passage_rows = []
    for test_index, heldout_work in enumerate(work_ids):
        train = np.ones(len(work_ids), dtype=bool)
        train[test_index] = False
        if not np.any(train & (composers == CHOPIN)) or not np.any(train & (composers == MOZART)):
            continue
        chopin_centroid = _normalize(work_matrix[train & (composers == CHOPIN)].mean(axis=0, keepdims=True))[0]
        mozart_centroid = _normalize(work_matrix[train & (composers == MOZART)].mean(axis=0, keepdims=True))[0]
        axis = _normalize((chopin_centroid - mozart_centroid)[None, :])[0]
        work_score = float(work_matrix[test_index] @ axis)
        work_rows.append(
            {
                "heldout_work_id": heldout_work,
                "composer": composers[test_index],
                "crossfit_axis_score": work_score,
                "predicted_composer": CHOPIN if work_score >= 0 else MOZART,
                "correct": bool((work_score >= 0 and composers[test_index] == CHOPIN) or (work_score < 0 and composers[test_index] == MOZART)),
            }
        )
        prompt_scores = prompt_matrix @ axis
        for prompt_index, prompt in prompts.iterrows():
            prompt_rows.append(
                {
                    "heldout_work_id": heldout_work,
                    "heldout_composer": composers[test_index],
                    "prompt_id": prompt["prompt_id"],
                    "family": prompt.get("family", ""),
                    "subfamily": prompt.get("subfamily", ""),
                    "prompt_text": prompt.get("prompt_text", ""),
                    "crossfit_axis_score": float(prompt_scores[prompt_index]),
                }
            )
        mask = passages["work_id"].to_numpy(str) == heldout_work
        scores = passage_matrix[mask] @ axis
        for passage_id, score in zip(passages.loc[mask, "passage_id"].astype(str), scores):
            passage_rows.append(
                {
                    "heldout_work_id": heldout_work,
                    "composer": composers[test_index],
                    "passage_id": passage_id,
                    "crossfit_axis_score": float(score),
                }
            )
    prompt_frame = pd.DataFrame(prompt_rows)
    work_frame = pd.DataFrame(work_rows)
    passage_frame = pd.DataFrame(passage_rows)
    prompt_summary = (
        prompt_frame.groupby(["prompt_id", "family", "subfamily", "prompt_text"], as_index=False)["crossfit_axis_score"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    prompt_summary.to_csv(output_tables / "crossfit_axis_prompt_projection.csv", index=False)
    work_frame.to_csv(output_tables / "crossfit_axis_work_projection.csv", index=False)
    passage_frame.to_csv(output_tables / "crossfit_axis_passage_projection.csv", index=False)
    return {
        "work_axis_accuracy": float(work_frame["correct"].mean()) if len(work_frame) else None,
        "work_axis_correct": int(work_frame["correct"].sum()) if len(work_frame) else 0,
        "work_axis_total": int(len(work_frame)),
    }


def _source_origin_sensitivity(passages: pd.DataFrame, matrix: np.ndarray, output_tables: Path) -> dict[str, Any]:
    rows = []
    for origin, origin_frame in passages.groupby("audio_origin", sort=True):
        if not origin or origin_frame["composer"].nunique() < 2:
            continue
        work_table, work_matrix = _work_table(origin_frame.reset_index(drop=True), matrix[origin_frame.index.to_numpy()])
        composers = work_table["composer"].to_numpy(str)
        if not np.any(composers == CHOPIN) or not np.any(composers == MOZART):
            continue
        centroids = {
            composer: _normalize(work_matrix[composers == composer].mean(axis=0, keepdims=True))[0]
            for composer in (CHOPIN, MOZART)
        }
        sims = work_matrix @ work_matrix.T
        left, right = np.triu_indices(len(work_table), k=1)
        within = composers[left] == composers[right]
        rows.append(
            {
                "audio_origin": origin,
                "work_count": int(len(work_table)),
                "passage_count": int(len(origin_frame)),
                "chopin_work_count": int(np.sum(composers == CHOPIN)),
                "mozart_work_count": int(np.sum(composers == MOZART)),
                "centroid_similarity": float(centroids[CHOPIN] @ centroids[MOZART]),
                "within_mean": float(np.mean(sims[left, right][within])),
                "between_mean": float(np.mean(sims[left, right][~within])),
                "within_minus_between": float(np.mean(sims[left, right][within]) - np.mean(sims[left, right][~within])),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_tables / "source_origin_work_sensitivity.csv", index=False)
    return {"available_origins": frame.to_dict(orient="records")}


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    tables = output / "tables"
    reports = output / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    prompt_matrix, prompts, prompt_metadata = _load_bundle(args.prompt_bundle)
    passage_matrix, passages, passage_metadata = _load_bundle(args.passage_bundle)
    work_table, work_matrix = _work_table(passages, passage_matrix)

    response_profiles = _prompt_response_profiles(prompt_matrix, prompts, passage_matrix, passages)
    response_profiles.to_csv(tables / "prompt_response_profiles.csv", index=False)
    clustering = _prompt_clustering(
        prompt_matrix,
        prompts,
        response_profiles,
        tables,
        min_k=args.min_clusters,
        max_k=args.max_clusters,
    )
    crossfit = _cross_fitted_axis(work_table, work_matrix, prompt_matrix, prompts, passages, passage_matrix, tables)
    source_origin = _source_origin_sensitivity(passages, passage_matrix, tables)

    report = {
        "prompt_bundle": str(Path(args.prompt_bundle).resolve()),
        "passage_bundle": str(Path(args.passage_bundle).resolve()),
        "output_dir": str(output.resolve()),
        "prompt_metadata": {
            "row_count": prompt_metadata.get("row_count"),
            "model_space": prompt_metadata.get("model_space"),
            "run_id": prompt_metadata.get("run_id"),
        },
        "passage_metadata": {
            "row_count": passage_metadata.get("row_count"),
            "model_space": passage_metadata.get("model_space"),
            "run_id": passage_metadata.get("run_id"),
        },
        "work_count": int(len(work_table)),
        "composer_work_counts": work_table["composer"].value_counts().to_dict(),
        "composer_passage_counts": passages["composer"].value_counts().to_dict(),
        "prompt_clustering": clustering,
        "crossfit_axis": crossfit,
        "source_origin_sensitivity": source_origin,
    }
    (reports / "advanced_prompt_embedding_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-bundle", required=True)
    parser.add_argument("--passage-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-clusters", type=int, default=3)
    parser.add_argument("--max-clusters", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run(args)
    print(
        json.dumps(
            {
                "output_dir": report["output_dir"],
                "prompt_clustering": report["prompt_clustering"],
                "crossfit_axis": report["crossfit_axis"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
