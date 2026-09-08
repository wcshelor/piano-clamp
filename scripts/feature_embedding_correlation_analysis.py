#!/usr/bin/env python3
"""Correlate transparent MusicXML features with CLaMP embedding geometry."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.embedding_io import read_embedding_bundle  # noqa: E402
from piano_clamp.features import FEATURE_FIELDS, FeatureError, extract_musicxml_features  # noqa: E402
from piano_clamp.prepare import PreparationError, crop_musicxml, mxl_musicxml  # noqa: E402


DEFAULT_METADATA_COLUMNS = [
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
    "start_bar",
    "end_bar",
]


def _normalize(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("cannot normalize zero-length embedding vectors")
    return array / norms


def _load_bundle(path: Path) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    matrix, rows, metadata = read_embedding_bundle(path)
    frame = pd.DataFrame(rows)
    if "embedding_row" not in frame:
        raise ValueError(f"embedding bundle table has no embedding_row column: {path}")
    frame = frame.loc[frame["embedding_row"].astype(str) != ""].copy()
    frame = frame.sort_values("embedding_row", key=lambda col: col.astype(int)).reset_index(drop=True)
    if len(frame) != matrix.shape[0]:
        raise ValueError(f"bundle table/matrix mismatch for {path}: {len(frame)} rows vs {matrix.shape[0]} vectors")
    return _normalize(matrix), frame, metadata


def _feature_columns(features: pd.DataFrame) -> list[str]:
    return [
        field
        for field in FEATURE_FIELDS
        if field in features.columns and pd.api.types.is_numeric_dtype(features[field])
    ]


def _safe_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(str(value)))
    except ValueError:
        return None


def _source_path(path_value: Any, corpus_root: Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else corpus_root / path


def _extract_window_features_from_bundle(
    passages: pd.DataFrame,
    *,
    corpus_root: Path,
    output_tables: Path,
) -> Path:
    if "score_path" not in passages.columns:
        raise ValueError("--corpus-root feature extraction requires score_path in the passage bundle")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    metadata_cols = [col for col in DEFAULT_METADATA_COLUMNS if col in passages.columns]
    with tempfile.TemporaryDirectory(prefix="feature-windows-", dir=output_tables.parent) as temporary_name:
        temporary = Path(temporary_name)
        for index, passage in passages.iterrows():
            passage_id = str(passage["passage_id"])
            try:
                source = _source_path(passage["score_path"], corpus_root)
                if not source.is_file():
                    raise FeatureError(f"source score does not exist: {source}")
                suffix = source.suffix.lower()
                if suffix not in {".xml", ".musicxml", ".mxl"}:
                    raise FeatureError(f"unsupported source score suffix for feature extraction: {source.suffix}")
                raw = mxl_musicxml(source.read_bytes()) if suffix == ".mxl" else source.read_bytes()
                start = _safe_int(passage.get("bar_start", passage.get("start_bar", None)))
                end = _safe_int(passage.get("bar_end", passage.get("end_bar", None)))
                if start is not None and end is not None and end >= start:
                    raw, _, _ = crop_musicxml(raw, start_index=start - 1, window_bars=end - start + 1)
                window_path = temporary / f"{index}.musicxml"
                window_path.write_bytes(raw)
                record: dict[str, Any] = {}
                for col in metadata_cols:
                    record[col] = passage.get(col, "")
                record.update(extract_musicxml_features(window_path))
                rows.append(record)
            except (FeatureError, PreparationError, OSError, KeyError, ValueError) as exc:
                failures.append({"passage_id": passage_id, "error_message": str(exc)})
    if not rows:
        raise ValueError("feature extraction from passage bundle produced no successful rows")
    feature_path = output_tables / "window_musicxml_features.csv"
    pd.DataFrame(rows).to_csv(feature_path, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_tables / "window_musicxml_feature_failures.csv", index=False)
    return feature_path


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    left = x[mask].astype(float)
    right = y[mask].astype(float)
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(float)


def _correlation_rows(
    predictors: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    predictor_label: str = "feature",
    outcome_label: str = "target",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for predictor in predictors.columns:
        x = predictors[predictor].to_numpy(float)
        for outcome in outcomes.columns:
            y = outcomes[outcome].to_numpy(float)
            rows.append(
                {
                    predictor_label: predictor,
                    outcome_label: outcome,
                    "n": int(np.sum(np.isfinite(x) & np.isfinite(y))),
                    "pearson_r": _pearson(x, y),
                    "spearman_r": _pearson(_rank(x), _rank(y)),
                }
            )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["abs_spearman_r"] = frame["spearman_r"].abs()
        frame["abs_pearson_r"] = frame["pearson_r"].abs()
    return frame


def _pca_scores(matrix: np.ndarray, components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centered = np.asarray(matrix, dtype=np.float64) - np.mean(matrix, axis=0, keepdims=True)
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    keep = min(components, vt.shape[0])
    scores = u[:, :keep] * singular[:keep]
    denom = max(matrix.shape[0] - 1, 1)
    eigenvalues = (singular**2) / denom
    explained = eigenvalues / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else np.zeros_like(eigenvalues)
    return scores, vt[:keep], eigenvalues[:keep], explained[:keep]


def _unit_table(
    passages: pd.DataFrame,
    matrix: np.ndarray,
    features: pd.DataFrame,
    feature_cols: list[str],
    unit: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if unit not in passages.columns:
        raise ValueError(f"unit column is missing from passage bundle: {unit}")
    if unit not in features.columns:
        if unit == "passage_id":
            features[unit] = features["passage_id"].astype(str)
        else:
            raise ValueError(f"unit column is missing from feature table: {unit}")

    records: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    feature_by_unit = features.groupby(unit, dropna=False)[feature_cols].mean(numeric_only=True)
    metadata_cols = [col for col in DEFAULT_METADATA_COLUMNS if col in passages.columns]
    for unit_id, group in passages.groupby(unit, sort=True, dropna=False):
        key = str(unit_id)
        if unit_id not in feature_by_unit.index:
            continue
        indices = group.index.to_numpy()
        vector = _normalize(matrix[indices].mean(axis=0, keepdims=True))[0]
        feature_values = feature_by_unit.loc[unit_id]
        if not np.isfinite(feature_values.to_numpy(float)).any():
            continue
        row: dict[str, Any] = {unit: key, "source_passage_count": int(len(group))}
        for col in metadata_cols:
            values = group[col].dropna().astype(str)
            row[col] = values.mode().iloc[0] if not values.empty else ""
        for feature in feature_cols:
            row[feature] = float(feature_values[feature])
        records.append(row)
        vectors.append(vector)
    if not records:
        raise ValueError("no aligned feature/embedding units were found")
    return pd.DataFrame(records), np.stack(vectors).astype(np.float32)


def _prompt_tables(
    prompt_bundle: Path | None,
    unit_table: pd.DataFrame,
    unit_matrix: np.ndarray,
    pca_components: np.ndarray,
    pc_names: list[str],
    unit: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if prompt_bundle is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    prompt_matrix, prompts, metadata = _load_bundle(prompt_bundle)
    if prompt_matrix.shape[1] != unit_matrix.shape[1]:
        raise ValueError(
            f"prompt and target embedding dimensions differ: {prompt_matrix.shape[1]} vs {unit_matrix.shape[1]}"
        )
    values = prompt_matrix @ unit_matrix.T
    prompt_similarity = pd.DataFrame({unit: unit_table[unit].astype(str)})
    for index, prompt in prompts.iterrows():
        prompt_similarity[str(prompt["prompt_id"])] = values[index]

    prompt_pc_projection = pd.DataFrame(prompt_matrix @ pca_components.T, columns=pc_names)
    prompt_pc_projection.insert(0, "prompt_id", prompts["prompt_id"].astype(str).to_numpy())
    for column in ("family", "subfamily", "polarity", "composer_reference", "interpretive_level", "prompt_text"):
        if column in prompts.columns:
            prompt_pc_projection.insert(
                len([col for col in prompt_pc_projection.columns if col in {"prompt_id", "family", "subfamily", "polarity", "composer_reference", "interpretive_level", "prompt_text"}]),
                column,
                prompts[column].astype(str).to_numpy(),
            )

    axis_scores = pd.DataFrame({unit: unit_table[unit].astype(str)})
    axis_manifest_rows: list[dict[str, Any]] = []
    family = prompts["family"] if "family" in prompts.columns else pd.Series([""] * len(prompts))
    polarity = prompts["polarity"] if "polarity" in prompts.columns else pd.Series([""] * len(prompts))
    matched = prompts.loc[
        (family == "matched_pair")
        & polarity.isin(["positive", "negative"])
    ].copy()
    if "subfamily" not in matched.columns:
        matched["subfamily"] = ""
    for axis_id, group in matched.groupby("subfamily", sort=True):
        poles = {row["polarity"]: int(row["embedding_row"]) for _, row in group.iterrows()}
        if set(poles) != {"positive", "negative"}:
            continue
        axis_scores[str(axis_id)] = values[poles["positive"]] - values[poles["negative"]]
        axis_manifest_rows.append(
            {
                "axis_id": axis_id,
                "positive_prompt_id": prompts.iloc[poles["positive"]]["prompt_id"],
                "negative_prompt_id": prompts.iloc[poles["negative"]]["prompt_id"],
            }
        )
    return prompt_similarity, axis_scores, pd.DataFrame(axis_manifest_rows), prompt_pc_projection, metadata


def _composer_axis(unit_table: pd.DataFrame, unit_matrix: np.ndarray) -> pd.DataFrame:
    if "composer" not in unit_table.columns or unit_table["composer"].nunique() != 2:
        return pd.DataFrame()
    composers = sorted(unit_table["composer"].dropna().astype(str).unique())
    centroids = {
        composer: _normalize(unit_matrix[unit_table["composer"].astype(str).to_numpy() == composer].mean(axis=0, keepdims=True))[0]
        for composer in composers
    }
    axis = _normalize((centroids[composers[0]] - centroids[composers[1]])[None, :])[0]
    return pd.DataFrame(
        {
            "composer_axis": unit_matrix @ axis,
            "composer_axis_positive": composers[0],
            "composer_axis_negative": composers[1],
        }
    )


def _write_outputs(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    tables = output / "tables"
    reports = output / "reports"
    tables.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    passage_matrix, passages, passage_metadata = _load_bundle(Path(args.passage_bundle))
    if args.feature_table:
        feature_table_path = Path(args.feature_table)
    else:
        if not args.corpus_root:
            raise ValueError("pass --feature-table, or pass --corpus-root to extract features from the passage bundle")
        feature_table_path = _extract_window_features_from_bundle(
            passages,
            corpus_root=Path(args.corpus_root),
            output_tables=tables,
        )
    features = pd.read_csv(feature_table_path)
    if "passage_id" not in features.columns:
        raise ValueError("feature table must contain passage_id")
    features["passage_id"] = features["passage_id"].astype(str)
    passages["passage_id"] = passages["passage_id"].astype(str)
    feature_cols = _feature_columns(features)
    if not feature_cols:
        raise ValueError("feature table contains none of the declared numeric MusicXML features")
    if args.unit != "passage_id":
        id_to_unit = passages.set_index("passage_id")[args.unit].astype(str)
        features[args.unit] = features["passage_id"].map(id_to_unit)
        features = features.loc[features[args.unit].notna()].copy()

    unit_table, unit_matrix = _unit_table(passages, passage_matrix, features, feature_cols, args.unit)
    unit_table.to_csv(tables / "aligned_feature_embedding_units.csv", index=False)

    feature_frame = unit_table[feature_cols].apply(pd.to_numeric, errors="coerce")
    pca_scores, pca_components, eigenvalues, explained = _pca_scores(unit_matrix, args.pca_components)
    pc_names = [f"pc{index + 1}" for index in range(pca_scores.shape[1])]
    pc_frame = pd.DataFrame(pca_scores, columns=pc_names)
    pd.concat([unit_table[[args.unit]].reset_index(drop=True), pc_frame], axis=1).to_csv(
        tables / "embedding_pc_scores.csv", index=False
    )
    pd.DataFrame(
        {
            "component": pc_names,
            "eigenvalue": eigenvalues,
            "explained_variance_ratio": explained,
            "cumulative_explained_variance_ratio": np.cumsum(explained),
        }
    ).to_csv(tables / "embedding_variance_spectrum.csv", index=False)

    feature_pc = _correlation_rows(feature_frame, pc_frame, outcome_label="embedding_axis")
    feature_pc.sort_values(["abs_spearman_r", "abs_pearson_r"], ascending=False).to_csv(
        tables / "feature_embedding_pc_correlations.csv", index=False
    )

    prompt_similarity, prompt_axes, axis_manifest, prompt_pc_projection, prompt_metadata = _prompt_tables(
        Path(args.prompt_bundle) if args.prompt_bundle else None,
        unit_table,
        unit_matrix,
        pca_components,
        pc_names,
        args.unit,
    )
    if not prompt_similarity.empty:
        prompt_similarity.to_csv(tables / "unit_prompt_similarities.csv", index=False)
        prompt_pc_projection.to_csv(tables / "prompt_embedding_pc_projection.csv", index=False)
        prompt_cols = [col for col in prompt_similarity.columns if col != args.unit]
        feature_prompt = _correlation_rows(feature_frame, prompt_similarity[prompt_cols], outcome_label="prompt_id")
        feature_prompt.sort_values(["abs_spearman_r", "abs_pearson_r"], ascending=False).to_csv(
            tables / "feature_prompt_similarity_correlations.csv", index=False
        )
        prompt_pc = pd.DataFrame(_normalize(prompt_similarity[prompt_cols].to_numpy(float)).T @ _normalize(pc_frame.to_numpy(float)), index=prompt_cols, columns=pc_names)
        prompt_pc.index.name = "prompt_id"
        prompt_pc.reset_index().to_csv(tables / "prompt_response_pc_projection.csv", index=False)
    if not prompt_axes.empty:
        prompt_axes.to_csv(tables / "unit_matched_prompt_axis_scores.csv", index=False)
        axis_manifest.to_csv(tables / "matched_prompt_axis_manifest.csv", index=False)
        axis_cols = [col for col in prompt_axes.columns if col != args.unit]
        if axis_cols:
            feature_axis = _correlation_rows(feature_frame, prompt_axes[axis_cols], outcome_label="prompt_axis")
            feature_axis.sort_values(["abs_spearman_r", "abs_pearson_r"], ascending=False).to_csv(
                tables / "feature_matched_prompt_axis_correlations.csv", index=False
            )

    composer_axis = _composer_axis(unit_table, unit_matrix)
    if not composer_axis.empty:
        composer_axis_out = pd.concat([unit_table[[args.unit, "composer"]].reset_index(drop=True), composer_axis], axis=1)
        composer_axis_out.to_csv(tables / "unit_composer_axis_scores.csv", index=False)
        feature_composer = _correlation_rows(feature_frame, composer_axis[["composer_axis"]], outcome_label="embedding_axis")
        feature_composer.sort_values(["abs_spearman_r", "abs_pearson_r"], ascending=False).to_csv(
            tables / "feature_composer_axis_correlations.csv", index=False
        )

    dim_count = min(args.embedding_dimensions, unit_matrix.shape[1])
    dimension_frame = pd.DataFrame(unit_matrix[:, :dim_count], columns=[f"dim_{index:04d}" for index in range(dim_count)])
    feature_dim = _correlation_rows(feature_frame, dimension_frame, outcome_label="embedding_dimension")
    feature_dim = feature_dim.sort_values(["abs_spearman_r", "abs_pearson_r"], ascending=False)
    feature_dim.groupby("feature", group_keys=False).head(args.top_dimensions_per_feature).to_csv(
        tables / "feature_embedding_dimension_top_correlations.csv", index=False
    )

    report = {
        "feature_table": str(feature_table_path.resolve()),
        "passage_bundle": str(Path(args.passage_bundle).resolve()),
        "prompt_bundle": str(Path(args.prompt_bundle).resolve()) if args.prompt_bundle else None,
        "output_dir": str(output.resolve()),
        "unit": args.unit,
        "aligned_unit_count": int(len(unit_table)),
        "feature_count": int(len(feature_cols)),
        "embedding_dimension": int(unit_matrix.shape[1]),
        "pca_components": int(len(pc_names)),
        "passage_metadata": {
            "row_count": passage_metadata.get("row_count"),
            "model_space": passage_metadata.get("model_space"),
            "run_id": passage_metadata.get("run_id"),
        },
        "prompt_metadata": {
            "row_count": prompt_metadata.get("row_count"),
            "model_space": prompt_metadata.get("model_space"),
            "run_id": prompt_metadata.get("run_id"),
        },
        "outputs": sorted(str(path.relative_to(output)) for path in tables.glob("*.csv")),
    }
    (reports / "feature_embedding_correlation_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-table", help="CSV from extract-features, usually tables/musicxml_features.csv")
    parser.add_argument("--corpus-root", help="Corpus root used to extract a matching feature table from passage score_path/bar ranges")
    parser.add_argument("--passage-bundle", required=True, help="Directory containing passage_embeddings.npy/csv")
    parser.add_argument("--prompt-bundle", help="Directory containing prompt_embeddings.npy/csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--unit", default="work_id", choices=["passage_id", "work_id", "composition_id", "movement_id", "recording_id"])
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--embedding-dimensions", type=int, default=768)
    parser.add_argument("--top-dimensions-per-feature", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = _write_outputs(args)
    print(json.dumps({"output_dir": report["output_dir"], "aligned_unit_count": report["aligned_unit_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
