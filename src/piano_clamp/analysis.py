"""Preregistered work-level summaries, inference, diagnostics, and figures."""

from __future__ import annotations

import hashlib
import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .features import FEATURE_FIELDS
from .prompts import load_prompts
from .similarity import load_named_embeddings


class AnalysisError(RuntimeError):
    """Raised when preregistered analysis inputs are missing or inconsistent."""


def _benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if not len(values):
        return []
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result.tolist()


def _hedges_g(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    denominator = len(left) + len(right) - 2
    pooled_variance = (
        (len(left) - 1) * np.var(left, ddof=1)
        + (len(right) - 1) * np.var(right, ddof=1)
    ) / denominator
    if pooled_variance <= 0:
        return 0.0 if np.mean(left) == np.mean(right) else float("nan")
    correction = 1 - 3 / (4 * (len(left) + len(right)) - 9)
    return float(correction * (np.mean(left) - np.mean(right)) / math.sqrt(pooled_variance))


def _permutation_p_value(
    left: np.ndarray,
    right: np.ndarray,
    *,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, int, str]:
    observed = abs(float(np.mean(left) - np.mean(right)))
    pooled = np.concatenate([left, right])
    left_count = len(left)
    total_combinations = math.comb(len(pooled), left_count)
    exceed = 0
    tested = 0
    if total_combinations <= iterations:
        for indices in combinations(range(len(pooled)), left_count):
            mask = np.zeros(len(pooled), dtype=bool)
            mask[list(indices)] = True
            statistic = abs(float(np.mean(pooled[mask]) - np.mean(pooled[~mask])))
            exceed += statistic >= observed - 1e-15
            tested += 1
        return exceed / tested, tested, "exact"
    for _ in range(iterations):
        permuted = rng.permutation(pooled)
        statistic = abs(float(np.mean(permuted[:left_count]) - np.mean(permuted[left_count:])))
        exceed += statistic >= observed - 1e-15
        tested += 1
    return (exceed + 1) / (tested + 1), tested, "monte_carlo"


def _work_level_contrast(
    frame: pd.DataFrame,
    value_column: str,
    composers: Sequence[str],
    *,
    group_column: str = "work_title",
    bootstrap_iterations: int,
    permutation_iterations: int,
    alpha: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if group_column not in frame.columns:
        raise AnalysisError(f"inference group column is missing: {group_column}")
    work_means = (
        frame.groupby(["composer", group_column], as_index=False)[value_column]
        .mean()
        .dropna(subset=[value_column])
    )
    left = work_means.loc[work_means["composer"] == composers[0], value_column].to_numpy(float)
    right = work_means.loc[work_means["composer"] == composers[1], value_column].to_numpy(float)
    if len(left) < 2 or len(right) < 2:
        raise AnalysisError(
            f"both composers need at least two independent works for {value_column}"
        )
    observed = float(np.mean(left) - np.mean(right))
    bootstrap = np.empty(bootstrap_iterations, dtype=float)
    for index in range(bootstrap_iterations):
        sampled_left = rng.choice(left, size=len(left), replace=True)
        sampled_right = rng.choice(right, size=len(right), replace=True)
        bootstrap[index] = np.mean(sampled_left) - np.mean(sampled_right)
    lower, upper = np.quantile(bootstrap, [alpha / 2, 1 - alpha / 2])
    p_value, permutations, method = _permutation_p_value(
        left,
        right,
        iterations=permutation_iterations,
        rng=rng,
    )
    return {
        "composer_a": composers[0],
        "composer_b": composers[1],
        "work_count_a": len(left),
        "work_count_b": len(right),
        "mean_a": float(np.mean(left)),
        "mean_b": float(np.mean(right)),
        "mean_difference_a_minus_b": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "hedges_g": _hedges_g(left, right),
        "permutation_p_value": p_value,
        "permutation_count": permutations,
        "permutation_method": method,
    }


def _feature_group_column(
    analysis_config: Mapping[str, Any],
    manifest: pd.DataFrame,
    features: pd.DataFrame,
) -> str:
    configured = str(analysis_config.get("feature_group_column", "")).strip()
    if configured:
        if configured not in manifest.columns or configured not in features.columns:
            raise AnalysisError(
                f"analysis.feature_group_column must exist in both manifest and feature table: {configured}"
            )
        return configured
    if "composition_id" in manifest.columns and "composition_id" in features.columns:
        return "composition_id"
    return "work_title"


def _write_feature_analysis(
    *,
    run_dir: Path,
    manifest: pd.DataFrame,
    feature_path: Path,
    composers: Sequence[str],
    analysis_config: Mapping[str, Any],
    bootstrap_iterations: int,
    permutation_iterations: int,
    alpha: float,
    seed: int,
) -> Path:
    if not feature_path.is_file():
        raise AnalysisError(f"preregistered MusicXML feature table is missing: {feature_path}")
    features = pd.read_csv(feature_path)
    required_feature_columns = {"passage_id", "composer", "work_title"}
    if missing := sorted(required_feature_columns - set(features.columns)):
        raise AnalysisError(f"MusicXML feature table is missing columns: {', '.join(missing)}")
    if features["passage_id"].duplicated().any():
        raise AnalysisError("MusicXML feature table contains duplicate passage IDs")
    if set(features["passage_id"].astype(str)) != set(manifest["passage_id"].astype(str)):
        raise AnalysisError("MusicXML feature passage coverage differs from the manifest")
    manifest_composers = manifest.set_index("passage_id")["composer"].astype(str)
    feature_composers = features.set_index("passage_id")["composer"].astype(str)
    if not feature_composers.sort_index().equals(manifest_composers.sort_index()):
        raise AnalysisError("MusicXML feature composer labels differ from the manifest")

    group_column = _feature_group_column(analysis_config, manifest, features)
    feature_contrasts = []
    numeric_features = [
        field
        for field in FEATURE_FIELDS
        if field in features and pd.api.types.is_numeric_dtype(features[field])
    ]
    rng = np.random.default_rng(seed + 1)
    for field in numeric_features:
        record: dict[str, Any] = {"feature": field, "inference_group": group_column}
        try:
            record.update(
                _work_level_contrast(
                    features,
                    field,
                    composers,
                    group_column=group_column,
                    bootstrap_iterations=bootstrap_iterations,
                    permutation_iterations=permutation_iterations,
                    alpha=alpha,
                    rng=rng,
                )
            )
            record["status"] = "ok"
            record["skip_reason"] = ""
        except AnalysisError as exc:
            record["status"] = "skipped"
            record["skip_reason"] = str(exc)
        feature_contrasts.append(record)
    feature_frame = pd.DataFrame(feature_contrasts)
    if feature_frame.empty:
        raise AnalysisError("MusicXML feature table contains none of the declared numeric features")
    feature_frame["fdr_q_value"] = np.nan
    valid_feature_mask = feature_frame["status"] == "ok"
    if not valid_feature_mask.any():
        raise AnalysisError("no MusicXML feature has adequate work-level data")
    feature_frame.loc[valid_feature_mask, "fdr_q_value"] = _benjamini_hochberg(
        feature_frame.loc[valid_feature_mask, "permutation_p_value"].tolist()
    )

    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    contrast_path = tables / "composer_feature_contrasts.csv"
    feature_frame.to_csv(contrast_path, index=False)
    (
        features.groupby("composer")[numeric_features]
        .agg(["count", "mean", "std", "median"])
        .to_csv(tables / "composer_feature_summary.csv")
    )
    plan_snapshot = {
        "analysis": dict(analysis_config),
        "contrast_interpretation": f"positive values mean {composers[0]} minus {composers[1]}",
        "inference_unit": f"{group_column}-level mean; bootstrap and permutation operate on groups",
        "random_seed": seed + 1,
        "model_free": True,
    }
    (run_dir / "feature_analysis_plan.resolved.yaml").write_text(
        yaml.safe_dump(plan_snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return contrast_path


def analyze_features(config: Mapping[str, Any]) -> Path:
    """Analyze deterministic MusicXML features without model embeddings or prompts."""

    run_dir = Path(config["paths"]["run_dir"])
    feature_path = run_dir / "tables" / "musicxml_features.csv"
    manifest_path = run_dir / "manifest.snapshot.csv"
    score_data_lock = run_dir / "inputs" / "score_data.lock.json"
    for required in (manifest_path, feature_path, score_data_lock):
        if not required.is_file():
            raise AnalysisError(f"feature analysis input is missing: {required}")

    analysis_config = config.get("analysis", {})
    composers = list(analysis_config.get("contrast_order", config["experiment"]["composers"]))
    if len(composers) != 2:
        raise AnalysisError("analysis.contrast_order must contain exactly two composers")
    manifest = pd.read_csv(manifest_path, dtype=str)
    required_manifest_columns = {"passage_id", "composer", "work_title"}
    if missing := sorted(required_manifest_columns - set(manifest.columns)):
        raise AnalysisError(f"manifest snapshot is missing columns: {', '.join(missing)}")
    return _write_feature_analysis(
        run_dir=run_dir,
        manifest=manifest,
        feature_path=feature_path,
        composers=composers,
        analysis_config=analysis_config,
        bootstrap_iterations=int(analysis_config.get("bootstrap_iterations", 5000)),
        permutation_iterations=int(analysis_config.get("permutation_iterations", 10000)),
        alpha=float(analysis_config.get("alpha", 0.05)),
        seed=int(analysis_config.get("random_seed", 20260819)),
    )


def _embedding_diagnostics(
    run_dir: Path,
    manifest: pd.DataFrame,
    *,
    max_pair_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ids = manifest["passage_id"].astype(str).tolist()
    matrix = load_named_embeddings(run_dir / "embeddings" / "scores", ids)
    norms = np.linalg.norm(matrix, axis=1)
    normalized = matrix / norms[:, None]
    hashes: dict[str, list[str]] = {}
    for passage_id, vector in zip(ids, matrix):
        digest = hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest()
        hashes.setdefault(digest, []).append(passage_id)
    duplicates = [members for members in hashes.values() if len(members) > 1]

    count = len(ids)
    pair_total = count * (count - 1) // 2
    if pair_total <= max_pair_samples:
        left_indices, right_indices = np.triu_indices(count, k=1)
    elif count > 1:
        left_indices = rng.integers(0, count, size=max_pair_samples)
        right_indices = rng.integers(0, count - 1, size=max_pair_samples)
        right_indices = right_indices + (right_indices >= left_indices)
    else:
        left_indices = right_indices = np.array([], dtype=int)
    similarities = np.sum(normalized[left_indices] * normalized[right_indices], axis=1)
    composers = manifest.set_index("passage_id").loc[ids, "composer"].to_numpy()
    within = composers[left_indices] == composers[right_indices]

    def describe(values: np.ndarray) -> dict[str, float | int | None]:
        return {
            "count": int(len(values)),
            "mean": float(np.mean(values)) if len(values) else None,
            "minimum": float(np.min(values)) if len(values) else None,
            "maximum": float(np.max(values)) if len(values) else None,
        }

    return {
        "embedding_count": count,
        "dimension": int(matrix.shape[1]),
        "all_finite": bool(np.isfinite(matrix).all()),
        "norm": describe(norms),
        "exact_duplicate_passage_groups": duplicates,
        "pair_sampling": {
            "possible_pairs": pair_total,
            "evaluated_pairs": int(len(similarities)),
            "within_composer": describe(similarities[within]),
            "between_composer": describe(similarities[~within]),
        },
    }


def _write_contrast_figure(
    contrasts: pd.DataFrame,
    prompt_lookup: Mapping[str, str],
    primary_prompt_ids: Sequence[str],
    output: Path,
) -> None:
    matplotlib_cache = output.parent.parent / "temp" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise AnalysisError("matplotlib is required to create preregistered figures") from exc
    selected = contrasts.set_index("prompt_id").loc[list(primary_prompt_ids)].reset_index()
    labels = [prompt_lookup[prompt_id] for prompt_id in selected["prompt_id"]]
    values = selected["mean_difference_a_minus_b"].to_numpy(float)
    lower = selected["ci_lower"].to_numpy(float)
    upper = selected["ci_upper"].to_numpy(float)
    y = np.arange(len(selected))
    figure, axis = plt.subplots(figsize=(9, max(4, 0.55 * len(selected))))
    axis.hlines(y, lower, upper, color="#2a9d8f", linewidth=1.5)
    axis.plot(values, y, "o", color="#264653")
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlabel("Work-level cosine-similarity difference (composer A − composer B)")
    axis.set_title("Preregistered CLaMP 3 C2 prompt contrasts (95% clustered bootstrap CI)")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def analyze(config: Mapping[str, Any]) -> Path:
    """Run the fixed, work-clustered analysis and write tables plus diagnostics."""

    run_dir = Path(config["paths"]["run_dir"])
    similarity_path = run_dir / "tables" / "prompt_similarity.csv"
    feature_path = run_dir / "tables" / "musicxml_features.csv"
    manifest_path = run_dir / "manifest.snapshot.csv"
    lock_paths = tuple(
        run_dir / "inputs" / f"{name}.lock.json"
        for name in ("score_data", "score_model", "prompt_data", "prompt_model")
    )
    for required in (similarity_path, manifest_path, *lock_paths):
        if not required.is_file():
            raise AnalysisError(f"analysis input is missing: {required}")

    analysis_config = config.get("analysis", {})
    composers = list(analysis_config.get("contrast_order", config["experiment"]["composers"]))
    if len(composers) != 2:
        raise AnalysisError("analysis.contrast_order must contain exactly two composers")
    primary_prompt_ids = list(analysis_config.get("primary_prompt_ids", []))
    if not primary_prompt_ids:
        raise AnalysisError("analysis.primary_prompt_ids must be preregistered")
    bootstrap_iterations = int(analysis_config.get("bootstrap_iterations", 5000))
    permutation_iterations = int(analysis_config.get("permutation_iterations", 10000))
    alpha = float(analysis_config.get("alpha", 0.05))
    seed = int(analysis_config.get("random_seed", 20260819))
    rng = np.random.default_rng(seed)

    manifest = pd.read_csv(manifest_path, dtype=str)
    similarities = pd.read_csv(similarity_path)
    manifest_columns = {"passage_id", "composer", "work_title", "movement"}
    similarity_columns = {
        "passage_id",
        "composer",
        "prompt_id",
        "prompt_text",
        "cosine_similarity",
    }
    if missing := sorted(manifest_columns - set(manifest.columns)):
        raise AnalysisError(f"manifest snapshot is missing columns: {', '.join(missing)}")
    if missing := sorted(similarity_columns - set(similarities.columns)):
        raise AnalysisError(f"prompt similarity table is missing columns: {', '.join(missing)}")
    similarities = similarities.merge(
        manifest[["passage_id", "composer", "work_title", "movement"]].rename(
            columns={"composer": "manifest_composer"}
        ),
        on="passage_id",
        how="left",
        validate="many_to_one",
    )
    if similarities["work_title"].isna().any():
        raise AnalysisError("similarity rows do not all match the manifest snapshot")
    if (similarities["composer"] != similarities["manifest_composer"]).any():
        raise AnalysisError("similarity composer labels do not match the manifest snapshot")
    if similarities.duplicated(["passage_id", "prompt_id"]).any():
        raise AnalysisError("prompt similarity table contains duplicate passage/prompt rows")
    if not np.isfinite(similarities["cosine_similarity"].to_numpy(float)).all():
        raise AnalysisError("prompt similarity table contains non-finite values")

    prompts = load_prompts(config["prompt_files"])
    prompt_lookup = {prompt["prompt_id"]: prompt["prompt_text"] for prompt in prompts}
    configured_prompt_ids = set(prompt_lookup)
    available_prompts = set(similarities["prompt_id"])
    if available_prompts != configured_prompt_ids:
        missing = sorted(configured_prompt_ids - available_prompts)
        unexpected = sorted(available_prompts - configured_prompt_ids)
        raise AnalysisError(
            "prompt coverage differs from configuration; "
            f"missing={missing}, unexpected={unexpected}"
        )
    expected_row_count = len(manifest) * len(configured_prompt_ids)
    if len(similarities) != expected_row_count:
        raise AnalysisError(
            f"prompt similarity table has {len(similarities)} rows; expected {expected_row_count}"
        )
    for prompt_id, prompt_text in prompt_lookup.items():
        observed_text = set(
            similarities.loc[similarities["prompt_id"] == prompt_id, "prompt_text"]
        )
        if observed_text != {prompt_text}:
            raise AnalysisError(f"stored wording differs from configured prompt: {prompt_id}")
    missing_prompts = sorted(set(primary_prompt_ids) - available_prompts)
    if missing_prompts:
        raise AnalysisError(f"preregistered prompts are missing: {', '.join(missing_prompts)}")

    summary = (
        similarities.groupby(["composer", "prompt_id", "prompt_text"], as_index=False)
        .agg(
            passage_count=("cosine_similarity", "size"),
            work_count=("work_title", "nunique"),
            mean=("cosine_similarity", "mean"),
            standard_deviation=("cosine_similarity", "std"),
            median=("cosine_similarity", "median"),
        )
    )
    prompt_contrasts = []
    for prompt_id, frame in similarities.groupby("prompt_id", sort=True):
        record = {"prompt_id": prompt_id, "prompt_text": frame["prompt_text"].iloc[0]}
        record.update(
            _work_level_contrast(
                frame,
                "cosine_similarity",
                composers,
                bootstrap_iterations=bootstrap_iterations,
                permutation_iterations=permutation_iterations,
                alpha=alpha,
                rng=rng,
            )
        )
        record["is_primary"] = prompt_id in primary_prompt_ids
        prompt_contrasts.append(record)
    contrast_frame = pd.DataFrame(prompt_contrasts)
    contrast_frame["fdr_q_value_primary"] = np.nan
    primary_mask = contrast_frame["is_primary"]
    contrast_frame.loc[primary_mask, "fdr_q_value_primary"] = _benjamini_hochberg(
        contrast_frame.loc[primary_mask, "permutation_p_value"].tolist()
    )

    tables = run_dir / "tables"
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "composer_prompt_summary.csv", index=False)
    contrast_frame.to_csv(tables / "composer_prompt_contrasts.csv", index=False)

    if analysis_config.get("require_features", True):
        _write_feature_analysis(
            run_dir=run_dir,
            manifest=manifest,
            feature_path=feature_path,
            composers=composers,
            analysis_config=analysis_config,
            bootstrap_iterations=bootstrap_iterations,
            permutation_iterations=permutation_iterations,
            alpha=alpha,
            seed=seed,
        )

    robustness_records = []
    indexed = contrast_frame.set_index("prompt_id")
    for pair in analysis_config.get("robustness_pairs", []):
        primary = pair["primary"]
        variant = pair["variant"]
        if primary not in indexed.index or variant not in indexed.index:
            raise AnalysisError(f"robustness pair is unavailable: {primary}, {variant}")
        robustness_records.append(
            {
                "primary_prompt_id": primary,
                "variant_prompt_id": variant,
                "primary_effect": indexed.loc[primary, "mean_difference_a_minus_b"],
                "variant_effect": indexed.loc[variant, "mean_difference_a_minus_b"],
                "absolute_effect_difference": abs(
                    indexed.loc[primary, "mean_difference_a_minus_b"]
                    - indexed.loc[variant, "mean_difference_a_minus_b"]
                ),
                "same_direction": bool(
                    np.sign(indexed.loc[primary, "mean_difference_a_minus_b"])
                    == np.sign(indexed.loc[variant, "mean_difference_a_minus_b"])
                ),
            }
        )
    pd.DataFrame(robustness_records).to_csv(tables / "prompt_robustness.csv", index=False)

    diagnostics = _embedding_diagnostics(
        run_dir,
        manifest,
        max_pair_samples=int(analysis_config.get("max_pair_samples", 100000)),
        rng=rng,
    )
    (tables / "embedding_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_contrast_figure(
        contrast_frame,
        prompt_lookup,
        primary_prompt_ids,
        figures / "primary_prompt_contrasts.png",
    )
    plan_snapshot = {
        "analysis": analysis_config,
        "contrast_interpretation": f"positive values mean {composers[0]} minus {composers[1]}",
        "inference_unit": "work-level mean; bootstrap and permutation operate on works",
        "random_seed": seed,
    }
    (run_dir / "analysis_plan.resolved.yaml").write_text(
        yaml.safe_dump(plan_snapshot, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return tables / "composer_prompt_contrasts.csv"
