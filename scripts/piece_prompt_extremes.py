#!/usr/bin/env python3
"""Find prompt-extreme works and cross-composer nearest pieces."""

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


# Generic multi-composer analysis. Legacy example used CHOPIN="Frédéric Chopin"
# and MOZART="Wolfgang Amadeus Mozart"; filtering is now explicit via
# --composer-a/--composer-b or all composers present are compared.


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


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""


def _work_vectors(passages: pd.DataFrame, matrix: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    _ensure_columns(
        passages,
        [
            "passage_id",
            "composer",
            "period",
            "composition_id",
            "work_id",
            "movement_id",
            "recording_id",
            "audio_origin",
            "passage_generation_method",
            "bar_start",
            "bar_end",
        ],
    )
    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    for work_id, group in passages.groupby("work_id", sort=True):
        indices = group.index.to_numpy()
        vectors.append(_normalize(matrix[indices].mean(axis=0, keepdims=True))[0])
        records.append(
            {
                "work_id": work_id,
                "composer": group["composer"].mode().iloc[0],
                "period": group["period"].mode().iloc[0] if not group["period"].dropna().empty else "",
                "composition_id": group["composition_id"].mode().iloc[0] if not group["composition_id"].dropna().empty else "",
                "passage_count": int(len(group)),
                "movement_count": int(group["movement_id"].nunique()),
                "recording_count": int(group["recording_id"].nunique()),
                "audio_origins": ";".join(sorted(group["audio_origin"].dropna().astype(str).unique())),
                "passage_methods": ";".join(sorted(group["passage_generation_method"].dropna().astype(str).unique())),
            }
        )
    return pd.DataFrame(records), np.stack(vectors).astype(np.float32)


def _prompt_work_scores(
    prompts: pd.DataFrame,
    prompt_matrix: np.ndarray,
    works: pd.DataFrame,
    work_matrix: np.ndarray,
) -> pd.DataFrame:
    values = prompt_matrix @ work_matrix.T
    rows = []
    for prompt_index, prompt in prompts.iterrows():
        frame = works.copy()
        frame["prompt_id"] = prompt["prompt_id"]
        frame["prompt_family"] = prompt.get("family", "")
        frame["prompt_subfamily"] = prompt.get("subfamily", "")
        frame["prompt_text"] = prompt.get("prompt_text", "")
        frame["mean_similarity"] = values[prompt_index]
        frame["rank_for_prompt"] = frame["mean_similarity"].rank(method="first", ascending=False).astype(int)
        frame["percentile_for_prompt"] = frame["mean_similarity"].rank(method="average", pct=True)
        rows.append(frame)
    output = pd.concat(rows, ignore_index=True)
    stats = output.groupby("prompt_id")["mean_similarity"].agg(["mean", "std"]).rename(
        columns={"mean": "prompt_work_mean", "std": "prompt_work_std"}
    )
    output = output.merge(stats, on="prompt_id", how="left")
    output["prompt_global_z"] = (
        (output["mean_similarity"] - output["prompt_work_mean"]) / output["prompt_work_std"].replace(0, np.nan)
    )
    composer_stats = output.groupby(["prompt_id", "composer"])["mean_similarity"].agg(["mean", "std"]).rename(
        columns={"mean": "prompt_composer_work_mean", "std": "prompt_composer_work_std"}
    )
    output = output.merge(composer_stats, on=["prompt_id", "composer"], how="left")
    output["prompt_composer_relative_z"] = (
        (output["mean_similarity"] - output["prompt_composer_work_mean"])
        / output["prompt_composer_work_std"].replace(0, np.nan)
    )
    return output


def _prompt_top_passages(
    prompts: pd.DataFrame,
    prompt_matrix: np.ndarray,
    passages: pd.DataFrame,
    passage_matrix: np.ndarray,
    prompt_top_works: pd.DataFrame,
    *,
    top_passages: int,
) -> pd.DataFrame:
    values = prompt_matrix @ passage_matrix.T
    prompt_index = {str(row["prompt_id"]): index for index, row in prompts.iterrows()}
    rows = []
    for _, match in prompt_top_works.iterrows():
        p_index = prompt_index[str(match["prompt_id"])]
        work_mask = passages["work_id"].astype(str).to_numpy() == str(match["work_id"])
        indices = np.flatnonzero(work_mask)
        if not len(indices):
            continue
        order = indices[np.argsort(-values[p_index, indices], kind="stable")[:top_passages]]
        for rank, passage_index in enumerate(order, start=1):
            passage = passages.iloc[int(passage_index)]
            rows.append(
                {
                    "prompt_id": match["prompt_id"],
                    "prompt_family": match["prompt_family"],
                    "prompt_subfamily": match["prompt_subfamily"],
                    "prompt_text": match["prompt_text"],
                    "work_id": match["work_id"],
                    "composer": match["composer"],
                    "work_rank_for_prompt": int(match["rank_for_prompt"]),
                    "passage_rank_within_work": rank,
                    "passage_id": passage["passage_id"],
                    "passage_similarity": float(values[p_index, passage_index]),
                    "bar_start": passage.get("bar_start", ""),
                    "bar_end": passage.get("bar_end", ""),
                    "recording_id": passage.get("recording_id", ""),
                    "audio_origin": passage.get("audio_origin", ""),
                    "passage_generation_method": passage.get("passage_generation_method", ""),
                }
            )
    return pd.DataFrame(rows)


def _work_top_prompts(prompt_work: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    return (
        prompt_work.sort_values(["work_id", "mean_similarity"], ascending=[True, False])
        .groupby("work_id", as_index=False, group_keys=False)
        .head(top_n)
        .copy()
    )


def _family_profiles(prompt_work: pd.DataFrame) -> pd.DataFrame:
    return (
        prompt_work.groupby(["work_id", "composer", "composition_id", "prompt_family"], as_index=False)["mean_similarity"]
        .mean()
        .pivot_table(
            index=["work_id", "composer", "composition_id"],
            columns="prompt_family",
            values="mean_similarity",
            aggfunc="first",
        )
        .reset_index()
    )


def _cross_composer_tables(
    works: pd.DataFrame,
    work_matrix: np.ndarray,
    *,
    composer_a: str | None = None,
    composer_b: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if works.empty or work_matrix.shape[0] == 0:
        empty = pd.DataFrame()
        return empty, empty, empty
    composers = works["composer"].astype(str).to_numpy()
    unique_composers = sorted(set(composers))

    # Optional explicit pair filter (e.g., legacy Chopin/Mozart).
    if composer_a is not None or composer_b is not None:
        if not composer_a or not composer_b:
            raise ValueError("--composer-a and --composer-b must be provided together")
        if composer_a == composer_b:
            raise ValueError("--composer-a and --composer-b must be distinct")
        requested = {composer_a, composer_b}
        mask = np.isin(composers, list(requested))
        if not np.any(mask):
            empty = pd.DataFrame()
            return empty, empty, empty
        # Restrict to the requested pair for nearest/centroid tables.
        filtered_indices = np.flatnonzero(mask)
        works_filtered = works.iloc[filtered_indices].reset_index(drop=True)
        matrix_filtered = work_matrix[filtered_indices]
        composers_filtered = works_filtered["composer"].astype(str).to_numpy()
        # Recurse with generic logic on the filtered subset without further pair args.
        return _cross_composer_tables(works_filtered, matrix_filtered)

    if len(unique_composers) < 2:
        empty = pd.DataFrame()
        return empty, empty, empty

    # Generic cross-composer nearest works: all unordered pairs with different composers.
    n = len(works)
    pair_rows: list[dict[str, object]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if composers[i] == composers[j]:
                continue
            left = works.iloc[int(i)]
            right = works.iloc[int(j)]
            sim = float(work_matrix[i] @ work_matrix[j])
            # Keep deterministic ordering by composer name for a/b assignment,
            # but preserve similarity value (symmetric).
            if str(left["composer"]) <= str(right["composer"]):
                a, b = left, right
            else:
                a, b = right, left
            pair_rows.append(
                {
                    "work_id_a": a["work_id"],
                    "composer_a": a["composer"],
                    "work_id_b": b["work_id"],
                    "composer_b": b["composer"],
                    "cosine_similarity": sim,
                    "passage_count_a": int(a["passage_count"]),
                    "passage_count_b": int(b["passage_count"]),
                }
            )
    if pair_rows:
        nearest_pairs = pd.DataFrame(pair_rows).sort_values("cosine_similarity", ascending=False).reset_index(drop=True)
        nearest_pairs["rank"] = np.arange(1, len(nearest_pairs) + 1)
    else:
        nearest_pairs = pd.DataFrame(pair_rows)

    # Generic centroid scores: compare each work to its own centroid vs best other centroid.
    composer_to_indices: dict[str, np.ndarray] = {
        c: np.flatnonzero(composers == c) for c in unique_composers
    }
    centroid_rows: list[dict[str, object]] = []
    for index, work in works.iterrows():
        own = str(work["composer"])
        own_members = np.flatnonzero((composers == own) & (np.arange(len(works)) != index))
        if len(own_members) == 0:
            continue
        own_centroid = _normalize(work_matrix[own_members].mean(axis=0, keepdims=True))[0]
        own_score = float(work_matrix[index] @ own_centroid)
        best_other: str | None = None
        best_score: float | None = None
        for other in unique_composers:
            if other == own:
                continue
            other_members = composer_to_indices[other]
            if len(other_members) == 0:
                continue
            other_centroid = _normalize(work_matrix[other_members].mean(axis=0, keepdims=True))[0]
            score = float(work_matrix[index] @ other_centroid)
            if best_score is None or score > best_score:
                best_score = score
                best_other = other
        if best_other is None or best_score is None:
            continue
        centroid_rows.append(
            {
                "work_id": work["work_id"],
                "composer": own,
                "opposite_composer": best_other,
                "similarity_to_own_composer_centroid": own_score,
                "similarity_to_opposite_composer_centroid": best_score,
                "own_minus_opposite_margin": own_score - best_score,
                "opposite_minus_own_score": best_score - own_score,
                "passage_count": int(work["passage_count"]),
                "recording_count": int(work["recording_count"]),
            }
        )
    if centroid_rows:
        centroid_scores = pd.DataFrame(centroid_rows).sort_values("own_minus_opposite_margin", ascending=True).reset_index(drop=True)
    else:
        centroid_scores = pd.DataFrame(centroid_rows)
    ambiguous = centroid_scores.head(min(20, len(centroid_scores))).copy() if len(centroid_scores) else centroid_scores.copy()
    return nearest_pairs, centroid_scores, ambiguous


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    tables = output / "tables"
    reports = output / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    prompt_matrix, prompts, prompt_metadata = _load_bundle(args.prompt_bundle)
    passage_matrix, passages, passage_metadata = _load_bundle(args.passage_bundle)
    works, work_matrix = _work_vectors(passages, passage_matrix)
    works.to_csv(tables / "work_manifest.csv", index=False)

    prompt_work = _prompt_work_scores(prompts, prompt_matrix, works, work_matrix)
    prompt_top_works = (
        prompt_work.sort_values(["prompt_id", "mean_similarity"], ascending=[True, False])
        .groupby("prompt_id", as_index=False, group_keys=False)
        .head(args.top_works)
        .copy()
    )
    prompt_outliers = prompt_work.loc[prompt_work["prompt_global_z"] >= args.outlier_z].sort_values(
        ["prompt_id", "prompt_global_z"], ascending=[True, False]
    )
    composer_outliers = prompt_work.loc[prompt_work["prompt_composer_relative_z"] >= args.outlier_z].sort_values(
        ["prompt_id", "composer", "prompt_composer_relative_z"], ascending=[True, True, False]
    )
    work_top_prompts = _work_top_prompts(prompt_work, top_n=args.top_prompts)
    family_profiles = _family_profiles(prompt_work)
    top_passages = _prompt_top_passages(
        prompts,
        prompt_matrix,
        passages,
        passage_matrix,
        prompt_top_works,
        top_passages=args.top_passages,
    )
    nearest_pairs, centroid_scores, ambiguous = _cross_composer_tables(
        works,
        work_matrix,
        composer_a=getattr(args, "composer_a", None),
        composer_b=getattr(args, "composer_b", None),
    )

    prompt_work.to_csv(tables / "prompt_work_scores.csv", index=False)
    prompt_top_works.to_csv(tables / "prompt_top_works.csv", index=False)
    prompt_outliers.to_csv(tables / "prompt_work_outliers.csv", index=False)
    composer_outliers.to_csv(tables / "composer_relative_prompt_work_outliers.csv", index=False)
    work_top_prompts.to_csv(tables / "work_top_prompts.csv", index=False)
    family_profiles.to_csv(tables / "work_prompt_family_profiles.csv", index=False)
    top_passages.to_csv(tables / "prompt_top_passages_within_top_works.csv", index=False)
    nearest_pairs.to_csv(tables / "cross_composer_nearest_works.csv", index=False)
    centroid_scores.to_csv(tables / "work_opposite_composer_centroid_scores.csv", index=False)
    ambiguous.to_csv(tables / "ambiguous_composer_works.csv", index=False)

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
        "work_count": int(len(works)),
        "prompt_count": int(len(prompts)),
        "composer_work_counts": works["composer"].value_counts().to_dict(),
        "composer_passage_counts": passages["composer"].value_counts().to_dict(),
        "row_counts": {
            "prompt_work_scores": int(len(prompt_work)),
            "prompt_top_works": int(len(prompt_top_works)),
            "prompt_work_outliers": int(len(prompt_outliers)),
            "composer_relative_prompt_work_outliers": int(len(composer_outliers)),
            "work_top_prompts": int(len(work_top_prompts)),
            "work_prompt_family_profiles": int(len(family_profiles)),
            "prompt_top_passages_within_top_works": int(len(top_passages)),
            "cross_composer_nearest_works": int(len(nearest_pairs)),
            "work_opposite_composer_centroid_scores": int(len(centroid_scores)),
            "ambiguous_composer_works": int(len(ambiguous)),
        },
        "settings": {
            "top_works": args.top_works,
            "top_prompts": args.top_prompts,
            "top_passages": args.top_passages,
            "outlier_z": args.outlier_z,
            "composer_a": getattr(args, "composer_a", None),
            "composer_b": getattr(args, "composer_b", None),
        },
    }
    if len(prompt_outliers):
        report["strongest_prompt_work_outliers"] = (
            prompt_outliers.sort_values("prompt_global_z", ascending=False)
            .head(10)[["prompt_id", "prompt_family", "prompt_subfamily", "work_id", "composer", "mean_similarity", "prompt_global_z"]]
            .to_dict(orient="records")
        )
    if len(nearest_pairs):
        report["nearest_cross_composer_pairs"] = nearest_pairs.head(10).to_dict(orient="records")
    if len(ambiguous):
        report["most_ambiguous_works"] = ambiguous.head(10).to_dict(orient="records")
    (reports / "piece_prompt_extremes_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-bundle", required=True)
    parser.add_argument("--passage-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-works", type=int, default=10)
    parser.add_argument("--top-prompts", type=int, default=10)
    parser.add_argument("--top-passages", type=int, default=5)
    parser.add_argument("--outlier-z", type=float, default=2.0)
    parser.add_argument(
        "--composer-a",
        default=None,
        help="Optional first composer for a filtered cross-composer pair analysis (requires --composer-b).",
    )
    parser.add_argument(
        "--composer-b",
        default=None,
        help="Optional second composer for a filtered cross-composer pair analysis (requires --composer-a).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run(args)
    print(
        json.dumps(
            {
                "output_dir": report["output_dir"],
                "row_counts": report["row_counts"],
                "nearest_cross_composer_pairs": report.get("nearest_cross_composer_pairs", [])[:3],
                "most_ambiguous_works": report.get("most_ambiguous_works", [])[:3],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
