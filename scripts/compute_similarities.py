#!/usr/bin/env python3
"""Compute same-checkpoint cosine analyses and matched-prompt axes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.data_loading import composer_matches, load_pipeline_config  # noqa: E402
from piano_clamp.embedding_io import EmbeddingIOError, read_embedding_bundle  # noqa: E402
from piano_clamp.similarity import cosine_similarity_matrix  # noqa: E402


TIDY_FIELDS = (
    "query_id",
    "target_id",
    "query_type",
    "target_type",
    "similarity",
    "composer",
    "period",
    "composition_id",
    "work_id",
    "movement_id",
    "recording_id",
    "audio_origin",
    "prompt_family",
    "prompt_subfamily",
    "model_space",
)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load(path: Path, item_type: str, warnings: list[str]) -> dict[str, Any] | None:
    try:
        matrix, rows, metadata = read_embedding_bundle(path)
    except EmbeddingIOError as exc:
        warnings.append(f"{item_type}: {exc}")
        return None
    successful = sorted(
        (row for row in rows if row.get("embedding_row", "") != ""),
        key=lambda row: int(row["embedding_row"]),
    )
    if len(successful) != matrix.shape[0]:
        warnings.append(f"{item_type}: table/matrix row mismatch")
        return None
    return {
        "matrix": np.asarray(matrix, dtype=np.float32),
        "rows": successful,
        "metadata": metadata,
        "type": item_type,
        "space": str(metadata.get("model_space", "unknown")),
        "id_field": str(metadata.get("id_field", "")),
    }


def _filtered(bundle: dict[str, Any], *, composer: str | None, family: str | None) -> dict[str, Any]:
    indices = []
    aliases = {"emotional": "emotion", "formal": "literal"}
    requested_family = aliases.get((family or "").casefold(), family)
    for index, row in enumerate(bundle["rows"]):
        if composer and not composer_matches(row.get("composer", ""), composer):
            continue
        if requested_family and row.get("family", "").casefold() != requested_family.casefold():
            continue
        indices.append(index)
    return {
        **bundle,
        "matrix": bundle["matrix"][indices],
        "rows": [bundle["rows"][index] for index in indices],
    }


def _cross_rows(prompts: dict[str, Any], targets: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray]:
    if prompts["space"] != targets["space"]:
        return [], np.empty((0, 0), dtype=np.float32)
    values = cosine_similarity_matrix(prompts["matrix"], targets["matrix"])
    rows: list[dict[str, Any]] = []
    for prompt_index, prompt in enumerate(prompts["rows"]):
        for target_index, target in enumerate(targets["rows"]):
            rows.append(
                {
                    "query_id": prompt["prompt_id"],
                    "target_id": target[targets["id_field"]],
                    "query_type": "prompt",
                    "target_type": targets["type"],
                    "similarity": float(values[prompt_index, target_index]),
                    "composer": target.get("composer", ""),
                    "period": target.get("period", ""),
                    "composition_id": target.get("composition_id", ""),
                    "work_id": target.get("work_id", ""),
                    "movement_id": target.get("movement_id", ""),
                    "recording_id": target.get("recording_id", ""),
                    "audio_origin": target.get("audio_origin", ""),
                    "prompt_family": prompt.get("family", ""),
                    "prompt_subfamily": prompt.get("subfamily", ""),
                    "model_space": prompts["space"],
                }
            )
    return rows, values


def _nearest_rows(
    prompts: dict[str, Any], targets: dict[str, Any], values: np.ndarray, top_k: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nearest_prompts: list[dict[str, Any]] = []
    nearest_targets: list[dict[str, Any]] = []
    if not values.size:
        return nearest_prompts, nearest_targets
    for target_index, target in enumerate(targets["rows"]):
        order = np.argsort(-values[:, target_index], kind="stable")[:top_k]
        for rank, prompt_index in enumerate(order, start=1):
            prompt = prompts["rows"][int(prompt_index)]
            nearest_prompts.append(
                {
                    "item_id": target[targets["id_field"]],
                    "item_type": targets["type"],
                    "neighbor_id": prompt["prompt_id"],
                    "neighbor_type": "prompt",
                    "rank": rank,
                    "similarity": float(values[prompt_index, target_index]),
                    "model_space": prompts["space"],
                }
            )
    for prompt_index, prompt in enumerate(prompts["rows"]):
        order = np.argsort(-values[prompt_index], kind="stable")[:top_k]
        for rank, target_index in enumerate(order, start=1):
            target = targets["rows"][int(target_index)]
            nearest_targets.append(
                {
                    "item_id": prompt["prompt_id"],
                    "item_type": "prompt",
                    "neighbor_id": target[targets["id_field"]],
                    "neighbor_type": targets["type"],
                    "rank": rank,
                    "similarity": float(values[prompt_index, target_index]),
                    "model_space": prompts["space"],
                }
            )
    return nearest_prompts, nearest_targets


def _composer_distances(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    mozart = [index for index, row in enumerate(bundle["rows"]) if composer_matches(row.get("composer", ""), "Mozart")]
    chopin = [index for index, row in enumerate(bundle["rows"]) if composer_matches(row.get("composer", ""), "Chopin")]
    if not mozart or not chopin:
        return []
    values = cosine_similarity_matrix(bundle["matrix"][mozart], bundle["matrix"][chopin])
    output = []
    for left_index, source_index in enumerate(mozart):
        source = bundle["rows"][source_index]
        for right_index, target_index in enumerate(chopin):
            target = bundle["rows"][target_index]
            similarity = float(values[left_index, right_index])
            output.append(
                {
                    "query_id": source[bundle["id_field"]],
                    "target_id": target[bundle["id_field"]],
                    "query_type": bundle["type"],
                    "target_type": bundle["type"],
                    "similarity": similarity,
                    "cosine_distance": 1.0 - similarity,
                    "query_composer": source.get("composer", ""),
                    "target_composer": target.get("composer", ""),
                    "model_space": bundle["space"],
                }
            )
    return output


def _axis_rows(prompts: dict[str, Any], targets: dict[str, Any], values: np.ndarray) -> list[dict[str, Any]]:
    by_axis: dict[str, dict[str, int]] = defaultdict(dict)
    for index, row in enumerate(prompts["rows"]):
        if row.get("family") == "matched_pair" and row.get("polarity") in {"positive", "negative"}:
            by_axis[row["subfamily"]][row["polarity"]] = index
    output = []
    for axis, poles in sorted(by_axis.items()):
        if set(poles) != {"positive", "negative"}:
            continue
        positive = poles["positive"]
        negative = poles["negative"]
        for target_index, target in enumerate(targets["rows"]):
            output.append(
                {
                    "target_id": target[targets["id_field"]],
                    "target_type": targets["type"],
                    "axis_id": axis,
                    "positive_prompt_id": prompts["rows"][positive]["prompt_id"],
                    "negative_prompt_id": prompts["rows"][negative]["prompt_id"],
                    "similarity_positive": float(values[positive, target_index]),
                    "similarity_negative": float(values[negative, target_index]),
                    "axis_score": float(values[positive, target_index] - values[negative, target_index]),
                    "composer": target.get("composer", ""),
                    "period": target.get("period", ""),
                    "composition_id": target.get("composition_id", ""),
                    "work_id": target.get("work_id", ""),
                    "movement_id": target.get("movement_id", ""),
                    "recording_id": target.get("recording_id", ""),
                    "audio_origin": target.get("audio_origin", ""),
                    "model_space": prompts["space"],
                }
            )
    return output


def _family_rows(prompts: dict[str, Any], targets: dict[str, Any], values: np.ndarray) -> list[dict[str, Any]]:
    families: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(prompts["rows"]):
        families[row.get("family", "")].append(index)
    output = []
    for family, indices in sorted(families.items()):
        for target_index, target in enumerate(targets["rows"]):
            output.append(
                {
                    "target_id": target[targets["id_field"]],
                    "target_type": targets["type"],
                    "prompt_family": family,
                    "mean_similarity": float(values[indices, target_index].mean()),
                    "prompt_count": len(indices),
                    "composer": target.get("composer", ""),
                    "composition_id": target.get("composition_id", ""),
                    "work_id": target.get("work_id", ""),
                    "movement_id": target.get("movement_id", ""),
                    "recording_id": target.get("recording_id", ""),
                    "audio_origin": target.get("audio_origin", ""),
                    "model_space": prompts["space"],
                }
            )
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_pipeline_config(args.config)
    root = Path(config["output_root"])
    analysis = Path(config["analysis_root"])
    similarities = analysis / "similarities"
    expected = [
        similarities / "prompt_to_passage.csv",
        similarities / "prompt_to_score.csv",
        similarities / "prompt_to_audio.csv",
        similarities / "mozart_chopin_passage_distances.csv",
        similarities / "mozart_chopin_score_distances.csv",
        similarities / "score_to_audio_correspondence.csv",
        similarities / "matched_prompt_axes.csv",
        similarities / "prompt_family_averages.csv",
    ]
    existing = [path for path in expected if path.exists()]
    if existing and not args.force:
        raise RuntimeError("refusing to overwrite similarity outputs; pass --force explicitly")
    warnings: list[str] = []
    bundles = {
        "text_c2": _load(root / "text", "prompt", warnings),
        "text_saas": _load(root / "text_saas", "prompt", warnings),
        "score": _load(root / "symbolic", "score", warnings),
        "audio": _load(root / "audio", "audio", warnings),
        "passage_score": _load(root / "passages", "passage_score", warnings),
        "passage_performance_midi": _load(
            root / "passages" / "performance_midi", "passage_performance_midi", warnings
        ),
        "passage_audio": _load(root / "passages" / "audio", "passage_audio", warnings),
    }
    for key, bundle in list(bundles.items()):
        if bundle:
            bundles[key] = _filtered(
                bundle,
                composer=args.composer if bundle["type"] != "prompt" else None,
                family=args.family if bundle["type"] == "prompt" else None,
            )

    prompt_to_score: list[dict[str, Any]] = []
    prompt_to_audio: list[dict[str, Any]] = []
    prompt_to_passage: list[dict[str, Any]] = []
    nearest_prompts: list[dict[str, Any]] = []
    nearest_passages: list[dict[str, Any]] = []
    axes: list[dict[str, Any]] = []
    families: list[dict[str, Any]] = []
    pairings = [
        ("text_c2", "score", prompt_to_score),
        ("text_saas", "audio", prompt_to_audio),
        ("text_c2", "passage_score", prompt_to_passage),
        ("text_c2", "passage_performance_midi", prompt_to_passage),
        ("text_saas", "passage_audio", prompt_to_passage),
    ]
    for prompt_key, target_key, destination in pairings:
        prompts, targets = bundles[prompt_key], bundles[target_key]
        if not prompts or not targets:
            continue
        if prompts["space"] != targets["space"]:
            warnings.append(
                f"skipped {prompt_key}->{target_key}: checkpoint spaces differ ({prompts['space']} vs {targets['space']})"
            )
            continue
        rows, values = _cross_rows(prompts, targets)
        destination.extend(rows)
        near_p, near_t = _nearest_rows(prompts, targets, values, args.top_k)
        nearest_prompts.extend(near_p)
        if "passage" in target_key:
            nearest_passages.extend(near_t)
        axes.extend(_axis_rows(prompts, targets, values))
        families.extend(_family_rows(prompts, targets, values))

    _atomic_csv(expected[0], prompt_to_passage, TIDY_FIELDS)
    _atomic_csv(expected[1], prompt_to_score, TIDY_FIELDS)
    _atomic_csv(expected[2], prompt_to_audio, TIDY_FIELDS)
    distance_fields = (
        "query_id", "target_id", "query_type", "target_type", "similarity", "cosine_distance",
        "query_composer", "target_composer", "model_space",
    )
    passage_distances = []
    for key in ("passage_score", "passage_performance_midi"):
        if bundles[key]:
            passage_distances.extend(_composer_distances(bundles[key]))
    score_distances = _composer_distances(bundles["score"]) if bundles["score"] else []
    _atomic_csv(expected[3], passage_distances, distance_fields)
    _atomic_csv(expected[4], score_distances, distance_fields)

    correspondence: list[dict[str, Any]] = []
    score_bundle, audio_bundle = bundles["score"], bundles["audio"]
    if score_bundle and audio_bundle:
        if score_bundle["space"] != audio_bundle["space"]:
            warnings.append("score/audio correspondence skipped: C2 and SAAS are separately trained geometries")
        else:
            audio_by_movement: dict[str, list[int]] = defaultdict(list)
            for index, row in enumerate(audio_bundle["rows"]):
                audio_by_movement[row.get("movement_id", "")].append(index)
            for score_index, score in enumerate(score_bundle["rows"]):
                for audio_index in audio_by_movement.get(score.get("movement_id", ""), []):
                    value = cosine_similarity_matrix(
                        score_bundle["matrix"][score_index : score_index + 1],
                        audio_bundle["matrix"][audio_index : audio_index + 1],
                    )[0, 0]
                    correspondence.append(
                        {
                            "score_id": score[score_bundle["id_field"]],
                            "recording_id": audio_bundle["rows"][audio_index][audio_bundle["id_field"]],
                            "movement_id": score.get("movement_id", ""),
                            "similarity": float(value),
                            "model_space": score_bundle["space"],
                        }
                    )
    _atomic_csv(expected[5], correspondence, ("score_id", "recording_id", "movement_id", "similarity", "model_space"))
    axis_fields = (
        "target_id", "target_type", "axis_id", "positive_prompt_id", "negative_prompt_id",
        "similarity_positive", "similarity_negative", "axis_score", "composer", "period",
        "composition_id", "work_id", "movement_id", "recording_id", "audio_origin", "model_space",
    )
    _atomic_csv(expected[6], axes, axis_fields)
    family_fields = (
        "target_id", "target_type", "prompt_family", "mean_similarity", "prompt_count", "composer",
        "composition_id", "work_id", "movement_id", "recording_id", "audio_origin", "model_space",
    )
    _atomic_csv(expected[7], families, family_fields)
    neighbor_fields = ("item_id", "item_type", "neighbor_id", "neighbor_type", "rank", "similarity", "model_space")
    _atomic_csv(analysis / "nearest_prompts" / "top_k.csv", nearest_prompts, neighbor_fields)
    _atomic_csv(analysis / "nearest_passages" / "top_k.csv", nearest_passages, neighbor_fields)
    report = {
        "top_k": args.top_k,
        "filters": {"composer": args.composer, "family": args.family},
        "row_counts": {path.name: sum(1 for _ in path.open(encoding="utf-8")) - 1 for path in expected},
        "warnings": warnings,
        "geometry_policy": "Cosine similarities are computed only when model_space values match exactly.",
    }
    report_path = analysis / "reports" / "similarity_run.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--composer")
    parser.add_argument("--family")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    try:
        result = run(args)
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
