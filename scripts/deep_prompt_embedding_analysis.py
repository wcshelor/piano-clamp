#!/usr/bin/env python3
"""Deep statistical summaries for prompt and composer passage embeddings."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

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


def _describe(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=float)
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


def _hedges_g(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    pooled_denominator = len(left) + len(right) - 2
    pooled_variance = (
        (len(left) - 1) * np.var(left, ddof=1)
        + (len(right) - 1) * np.var(right, ddof=1)
    ) / pooled_denominator
    if pooled_variance <= 0:
        return 0.0 if np.mean(left) == np.mean(right) else float("nan")
    correction = 1 - 3 / (4 * (len(left) + len(right)) - 9)
    return float(correction * (np.mean(left) - np.mean(right)) / math.sqrt(pooled_variance))


def _permutation_p(left: np.ndarray, right: np.ndarray, *, iterations: int, rng: np.random.Generator) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    observed = abs(float(np.mean(left) - np.mean(right)))
    pooled = np.concatenate([left, right])
    left_count = len(left)
    exceed = 0
    for _ in range(iterations):
        permuted = rng.permutation(pooled)
        statistic = abs(float(np.mean(permuted[:left_count]) - np.mean(permuted[left_count:])))
        exceed += statistic >= observed - 1e-15
    return float((exceed + 1) / (iterations + 1))


def _bootstrap_ci(
    left: np.ndarray,
    right: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if len(left) < 2 or len(right) < 2:
        return float("nan"), float("nan")
    values = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled_left = rng.choice(left, size=len(left), replace=True)
        sampled_right = rng.choice(right, size=len(right), replace=True)
        values[index] = np.mean(sampled_left) - np.mean(sampled_right)
    lower, upper = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def _bh_q_values(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    mask = np.isfinite(values)
    if not np.any(mask):
        return result.tolist()
    finite = values[mask]
    order = np.argsort(finite)
    ranked = finite[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    result[mask] = restored
    return result.tolist()


def _contrast_rows(
    frame: pd.DataFrame,
    *,
    key_columns: list[str],
    value_column: str,
    group_column: str,
    rng: np.random.Generator,
    bootstrap_iterations: int,
    permutation_iterations: int,
) -> pd.DataFrame:
    grouped = (
        frame.groupby(key_columns + ["composer", group_column], dropna=False, as_index=False)[value_column]
        .mean()
    )
    rows: list[dict[str, Any]] = []
    for keys, group in grouped.groupby(key_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(key_columns, keys))
        left = group.loc[group["composer"] == CHOPIN, value_column].to_numpy(float)
        right = group.loc[group["composer"] == MOZART, value_column].to_numpy(float)
        lower, upper = _bootstrap_ci(left, right, iterations=bootstrap_iterations, rng=rng)
        p_value = _permutation_p(left, right, iterations=permutation_iterations, rng=rng)
        record.update(
            {
                "inference_unit": group_column,
                "chopin_n": int(len(left)),
                "mozart_n": int(len(right)),
                "chopin_mean": float(np.mean(left)) if len(left) else float("nan"),
                "mozart_mean": float(np.mean(right)) if len(right) else float("nan"),
                "chopin_minus_mozart": float(np.mean(left) - np.mean(right)) if len(left) and len(right) else float("nan"),
                "ci_lower": lower,
                "ci_upper": upper,
                "hedges_g": _hedges_g(left, right),
                "permutation_p_value": p_value,
            }
        )
        rows.append(record)
    output = pd.DataFrame(rows)
    if not output.empty:
        output["fdr_q_value"] = _bh_q_values(output["permutation_p_value"].tolist())
    return output


def _sample_pair_geometry(
    passages: pd.DataFrame,
    matrix: np.ndarray,
    *,
    sample_pairs: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    count = matrix.shape[0]
    if count < 2:
        return {}
    left = rng.integers(0, count, size=sample_pairs)
    right = rng.integers(0, count - 1, size=sample_pairs)
    right = right + (right >= left)
    values = np.sum(matrix[left] * matrix[right], axis=1)
    composers = passages["composer"].to_numpy(str)
    kinds = np.where(composers[left] == composers[right], "within", "between")
    result = {
        "sample_pairs": int(sample_pairs),
        "all": _describe(values),
        "within_composer": _describe(values[kinds == "within"]),
        "between_composer": _describe(values[kinds == "between"]),
    }
    for composer in sorted(set(composers)):
        mask = (composers[left] == composer) & (composers[right] == composer)
        result[f"within_{composer}"] = _describe(values[mask])
    return result


def _nearest_neighbor_purity(
    passages: pd.DataFrame,
    matrix: np.ndarray,
    *,
    sample_size: int,
    block_size: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    count = matrix.shape[0]
    sample = np.arange(count) if count <= sample_size else np.sort(rng.choice(count, size=sample_size, replace=False))
    rows = []
    composers = passages["composer"].to_numpy(str)
    passage_ids = passages["passage_id"].to_numpy(str)
    work_ids = passages["work_id"].to_numpy(str)
    for start in range(0, len(sample), block_size):
        batch_indices = sample[start : start + block_size]
        sims = matrix[batch_indices] @ matrix.T
        sims[np.arange(len(batch_indices)), batch_indices] = -np.inf
        nearest = np.argmax(sims, axis=1)
        for source_index, target_index in zip(batch_indices, nearest):
            rows.append(
                {
                    "passage_id": passage_ids[source_index],
                    "composer": composers[source_index],
                    "work_id": work_ids[source_index],
                    "nearest_passage_id": passage_ids[target_index],
                    "nearest_composer": composers[target_index],
                    "nearest_work_id": work_ids[target_index],
                    "nearest_similarity": float(sims[list(batch_indices).index(source_index), target_index]),
                    "same_composer": bool(composers[source_index] == composers[target_index]),
                    "same_work": bool(work_ids[source_index] == work_ids[target_index]),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    tables = output / "tables"
    reports = output / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    prompt_matrix, prompt_rows, prompt_metadata = read_embedding_bundle(args.prompt_bundle)
    passage_matrix, passage_rows, passage_metadata = read_embedding_bundle(args.passage_bundle)
    prompts = pd.DataFrame(prompt_rows)
    passages = pd.DataFrame(passage_rows)
    prompts = prompts.loc[prompts["embedding_row"].astype(str) != ""].copy()
    passages = passages.loc[passages["embedding_row"].astype(str) != ""].copy()
    prompts = prompts.sort_values("embedding_row", key=lambda col: col.astype(int)).reset_index(drop=True)
    passages = passages.sort_values("embedding_row", key=lambda col: col.astype(int)).reset_index(drop=True)
    prompt_matrix = _normalize(prompt_matrix)
    passage_matrix = _normalize(passage_matrix)

    values = prompt_matrix @ passage_matrix.T
    long_rows = []
    for prompt_index, prompt in prompts.iterrows():
        frame = passages[
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
            ]
        ].copy()
        frame["prompt_id"] = prompt["prompt_id"]
        frame["prompt_family"] = prompt.get("family", "")
        frame["prompt_subfamily"] = prompt.get("subfamily", "")
        frame["prompt_text"] = prompt.get("prompt_text", "")
        frame["similarity"] = values[prompt_index]
        long_rows.append(frame)
    prompt_to_passage = pd.concat(long_rows, ignore_index=True)

    prompt_summary = (
        prompt_to_passage.groupby(["prompt_id", "prompt_family", "prompt_subfamily", "composer"], as_index=False)["similarity"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    prompt_summary.to_csv(tables / "prompt_by_composer_summary.csv", index=False)

    prompt_contrasts = _contrast_rows(
        prompt_to_passage,
        key_columns=["prompt_id", "prompt_family", "prompt_subfamily"],
        value_column="similarity",
        group_column=args.inference_unit,
        rng=rng,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
    ).sort_values("chopin_minus_mozart", ascending=False)
    prompt_contrasts.to_csv(tables / "prompt_composer_contrasts.csv", index=False)

    family_summary = (
        prompt_to_passage.groupby(["prompt_family", "composer"], as_index=False)["similarity"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
    )
    family_summary.to_csv(tables / "prompt_family_composer_summary.csv", index=False)
    family_contrasts = _contrast_rows(
        prompt_to_passage,
        key_columns=["prompt_family"],
        value_column="similarity",
        group_column=args.inference_unit,
        rng=rng,
        bootstrap_iterations=args.bootstrap_iterations,
        permutation_iterations=args.permutation_iterations,
    ).sort_values("chopin_minus_mozart", ascending=False)
    family_contrasts.to_csv(tables / "prompt_family_composer_contrasts.csv", index=False)

    axes = []
    matched = prompts.loc[(prompts.get("family", "") == "matched_pair") & prompts.get("polarity", "").isin(["positive", "negative"])]
    for axis_id, axis_prompts in matched.groupby("subfamily", sort=True):
        poles = {row["polarity"]: int(row["embedding_row"]) for _, row in axis_prompts.iterrows()}
        if set(poles) != {"positive", "negative"}:
            continue
        axis_score = values[poles["positive"]] - values[poles["negative"]]
        frame = passages[["passage_id", "composer", "work_id", "movement_id", "recording_id", "audio_origin"]].copy()
        frame["axis_id"] = axis_id
        frame["axis_score"] = axis_score
        axes.append(frame)
    if axes:
        axis_frame = pd.concat(axes, ignore_index=True)
        axis_summary = (
            axis_frame.groupby(["axis_id", "composer"], as_index=False)["axis_score"]
            .agg(["count", "mean", "std", "median", "min", "max"])
            .reset_index()
        )
        axis_summary.to_csv(tables / "matched_axis_composer_summary.csv", index=False)
        axis_contrasts = _contrast_rows(
            axis_frame,
            key_columns=["axis_id"],
            value_column="axis_score",
            group_column=args.inference_unit,
            rng=rng,
            bootstrap_iterations=args.bootstrap_iterations,
            permutation_iterations=args.permutation_iterations,
        ).sort_values("chopin_minus_mozart", ascending=False)
        axis_contrasts.to_csv(tables / "matched_axis_composer_contrasts.csv", index=False)

    by_composer = {
        composer: passage_matrix[passages["composer"].to_numpy(str) == composer]
        for composer in sorted(passages["composer"].dropna().unique())
    }
    centroids = {composer: _normalize(matrix.mean(axis=0, keepdims=True))[0] for composer, matrix in by_composer.items()}
    centroid_rows = []
    for prompt_index, prompt in prompts.iterrows():
        row = {
            "prompt_id": prompt["prompt_id"],
            "prompt_family": prompt.get("family", ""),
            "prompt_subfamily": prompt.get("subfamily", ""),
            "prompt_text": prompt.get("prompt_text", ""),
        }
        for composer, centroid in centroids.items():
            row[f"similarity_to_{composer}"] = float(prompt_matrix[prompt_index] @ centroid)
        if CHOPIN in centroids and MOZART in centroids:
            row["similarity_to_chopin_minus_mozart"] = row[f"similarity_to_{CHOPIN}"] - row[f"similarity_to_{MOZART}"]
        centroid_rows.append(row)
    prompt_centroids = pd.DataFrame(centroid_rows).sort_values("similarity_to_chopin_minus_mozart", ascending=False)
    prompt_centroids.to_csv(tables / "prompt_centroid_axis_projection.csv", index=False)

    nn = _nearest_neighbor_purity(
        passages,
        passage_matrix,
        sample_size=args.nearest_sample,
        block_size=args.block_size,
        rng=rng,
    )
    nn.to_csv(tables / "sampled_nearest_neighbor_purity.csv", index=False)

    geometry = {
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
        "composer_counts": passages["composer"].value_counts().to_dict(),
        "passage_methods": passages["passage_generation_method"].value_counts().to_dict(),
        "centroid_cosine_similarity": {
            f"{left}__{right}": float(centroids[left] @ centroids[right])
            for left in centroids
            for right in centroids
            if left < right
        },
        "sampled_pair_geometry": _sample_pair_geometry(passages, passage_matrix, sample_pairs=args.sample_pairs, rng=rng),
        "nearest_neighbor_purity": {
            "sampled_rows": int(len(nn)),
            "same_composer_rate": float(nn["same_composer"].mean()) if len(nn) else None,
            "same_work_rate": float(nn["same_work"].mean()) if len(nn) else None,
            "by_composer": nn.groupby("composer")["same_composer"].mean().to_dict() if len(nn) else {},
        },
        "statistical_settings": {
            "inference_unit": args.inference_unit,
            "bootstrap_iterations": args.bootstrap_iterations,
            "permutation_iterations": args.permutation_iterations,
            "seed": args.seed,
        },
    }
    (reports / "embedding_geometry_summary.json").write_text(
        json.dumps(geometry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return geometry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-bundle", required=True, help="Directory containing prompt_embeddings.npy/csv and metadata.json")
    parser.add_argument("--passage-bundle", required=True, help="Directory containing passage_embeddings.npy/csv and metadata.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inference-unit", default="work_id", choices=["work_id", "recording_id", "movement_id"])
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--permutation-iterations", type=int, default=5000)
    parser.add_argument("--sample-pairs", type=int, default=1_000_000)
    parser.add_argument("--nearest-sample", type=int, default=5000)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({"output_dir": result["output_dir"], "composer_counts": result["composer_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
