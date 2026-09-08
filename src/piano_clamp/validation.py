"""GPU-free validation and descriptive audit of an external passage manifest."""

from __future__ import annotations

import json
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

import numpy as np

from .embedding_io import sha256_path

from .data import read_manifest, resolve_source_files, write_manifest_snapshot
from .features import FeatureError, MUSICXML_EXTENSIONS, extract_musicxml_features


class DataValidationError(RuntimeError):
    """Raised after a structured validation report records blocking errors."""


def _describe(values: list[int]) -> dict[str, float | int]:
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": float(mean(values)),
        "median": float(median(values)),
    }


def validate_data(config: Mapping[str, Any]) -> Path:
    """Validate source paths and write a structured preflight report."""

    rows = read_manifest(config["paths"]["manifest"])
    sources = resolve_source_files(rows, config["paths"]["data_root"])
    from .embeddings import initialize_run

    run_dir = initialize_run(config)
    tables = run_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    write_manifest_snapshot(rows, run_dir / "manifest.snapshot.csv")

    errors: list[str] = []
    warnings: list[str] = []
    data_config = config.get("data", {})
    if data_config.get("passages_presegmented") is not True:
        errors.append(
            "data.passages_presegmented must be true: this pipeline does not crop bar ranges "
            "from complete scores"
        )

    relative_paths = [row["relative_path"] for row in rows]
    repeated_paths = sorted(path for path, count in Counter(relative_paths).items() if count > 1)
    if repeated_paths:
        errors.append(
            "multiple passage IDs reference the same source file; provide one pre-segmented file "
            f"per passage: {', '.join(repeated_paths[:10])}"
        )

    empty_files = [item_id for item_id, path in sources.items() if path.stat().st_size == 0]
    if empty_files:
        errors.append(f"empty source files: {', '.join(empty_files)}")

    composer_counts = Counter(row["composer"] for row in rows)
    work_passage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        work_passage_counts[row["composer"]][row["work_title"]] += 1

    expected_composers = list(config.get("experiment", {}).get("composers", []))
    expected_set = set(expected_composers)
    actual_set = set(composer_counts)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing:
        errors.append(f"configured composers absent from manifest: {', '.join(missing)}")
    if unexpected:
        errors.append(f"manifest contains unconfigured composers: {', '.join(unexpected)}")

    counts = list(composer_counts.values())
    if len(counts) > 1 and min(counts) and max(counts) / min(counts) > 1.5:
        warnings.append("passage counts differ by more than 50% across composers")
    for composer, works in work_passage_counts.items():
        if len(works) < 2:
            errors.append(f"{composer} has fewer than two independent works")
        dominant = max(works.values()) / sum(works.values())
        if dominant > 0.5:
            warnings.append(f"more than half of {composer} passages come from one work")

    bar_lengths = [int(row["end_bar"]) - int(row["start_bar"]) + 1 for row in rows]
    measure_count_mismatches = []
    if data_config.get("check_musicxml_parse", True):
        tolerance = int(data_config.get("measure_count_tolerance", 1))
        for row in rows:
            source = sources[row["passage_id"]]
            if source.suffix.lower() not in MUSICXML_EXTENSIONS:
                continue
            try:
                actual = int(extract_musicxml_features(source)["measure_count"])
            except FeatureError as exc:
                errors.append(str(exc))
                continue
            expected = int(row["end_bar"]) - int(row["start_bar"]) + 1
            if abs(actual - expected) > tolerance:
                measure_count_mismatches.append(
                    {
                        "passage_id": row["passage_id"],
                        "manifest_bar_span": expected,
                        "musicxml_measure_count": actual,
                    }
                )
        if measure_count_mismatches:
            warnings.append(
                "MusicXML measure counts differ from manifest bar spans beyond the configured "
                "tolerance; inspect data_validation.json"
            )
    extension_counts = Counter(path.suffix.lower() for path in sources.values())
    report = {
        "schema_version": 1,
        "status": "error" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "manifest": str(config["paths"]["manifest"]),
        "passages_presegmented": data_config.get("passages_presegmented"),
        "passage_count": len(rows),
        "composer_counts": dict(sorted(composer_counts.items())),
        "work_counts": {
            composer: len(works) for composer, works in sorted(work_passage_counts.items())
        },
        "passages_per_work": {
            composer: dict(sorted(works.items()))
            for composer, works in sorted(work_passage_counts.items())
        },
        "bar_span": _describe(bar_lengths),
        "format_counts": dict(sorted(extension_counts.items())),
        "total_source_bytes": sum(path.stat().st_size for path in sources.values()),
        "repeated_relative_paths": repeated_paths,
        "measure_count_mismatches": measure_count_mismatches,
    }
    output = tables / "data_validation.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if errors:
        raise DataValidationError(f"data validation failed; see {output}")
    return output


def validate_embedding_outputs(config: Mapping[str, Any]) -> dict[str, Any]:
    """Audit matrix/table/metadata alignment for every requested modality."""

    output_root = Path(str(config["output_root"]))
    analysis_root = Path(str(config["analysis_root"]))
    specifications = {
        "text": (output_root / "text", "prompt_id", True),
        "symbolic": (output_root / "symbolic", "score_id", True),
        "audio": (output_root / "audio", "recording_id", True),
        "passages": (output_root / "passages", "passage_id", True),
        "passages_performance_midi": (output_root / "passages" / "performance_midi", "passage_id", False),
        "passages_audio": (output_root / "passages" / "audio", "passage_id", False),
    }
    bundles: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []
    dimensions: dict[str, set[int]] = defaultdict(set)
    total_failures = 0
    for name, (directory, expected_id, required) in specifications.items():
        metadata_path = directory / "metadata.json"
        if not metadata_path.is_file():
            if required:
                errors.append(f"{name}: missing {metadata_path}")
            else:
                warnings.append(f"{name}: missing {metadata_path}")
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{name}: unreadable metadata: {exc}")
            continue
        matrix_path = directory / str(metadata.get("matrix_file", ""))
        table_path = directory / str(metadata.get("table_file", ""))
        bundle_errors: list[str] = []
        bundle_warnings: list[str] = []
        if not matrix_path.is_file():
            bundle_errors.append(f"missing matrix {matrix_path.name}")
        if not table_path.is_file():
            bundle_errors.append(f"missing table {table_path.name}")
        if bundle_errors:
            errors.extend(f"{name}: {value}" for value in bundle_errors)
            continue
        try:
            matrix = np.load(matrix_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: cannot load matrix: {exc}")
            continue
        with table_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if matrix.ndim != 2:
            bundle_errors.append(f"matrix is {matrix.ndim}-D rather than 2-D")
        elif not np.isfinite(matrix).all():
            bundle_errors.append("matrix contains non-finite values")
        else:
            dimensions[str(metadata.get("model_space", "unknown"))].add(int(matrix.shape[1]))
        id_field = str(metadata.get("id_field", ""))
        if id_field != expected_id:
            bundle_errors.append(f"id_field is {id_field!r}; expected {expected_id!r}")
        ids = [row.get(id_field, "") for row in rows]
        if not all(ids):
            bundle_errors.append("one or more IDs are empty")
        if len(ids) != len(set(ids)):
            bundle_errors.append("IDs are not unique")
        successful = [row for row in rows if row.get("embedding_row", "") != ""]
        if matrix.ndim == 2 and len(successful) != matrix.shape[0]:
            bundle_errors.append("successful metadata rows do not match matrix rows")
        try:
            indices = [int(row["embedding_row"]) for row in successful]
        except ValueError:
            bundle_errors.append("embedding_row contains a non-integer value")
        else:
            if matrix.ndim == 2 and indices != list(range(matrix.shape[0])):
                bundle_errors.append("embedding_row is not contiguous in matrix order")
        for required in ("model", "model_space", "checkpoint_path", "checkpoint_hash", "normalization"):
            if not metadata.get(required):
                bundle_errors.append(f"metadata is missing {required}")
        if not metadata.get("overwrite_guard"):
            bundle_errors.append("metadata does not record the overwrite guard")
        actual_hash = sha256_path(matrix_path)
        if metadata.get("embedding_sha256") != actual_hash:
            bundle_errors.append("matrix hash differs from metadata (determinism/integrity check failed)")
        source_field = {
            "symbolic": "source_path",
            "audio": "audio_path",
            "passages": "source_path",
            "passages_performance_midi": "source_path",
            "passages_audio": "source_path",
        }.get(name)
        if source_field:
            def has_source(row: Mapping[str, str]) -> bool:
                if row.get(source_field, ""):
                    return True
                material = row.get("source_material") or row.get("embedding_modality", "")
                if material == "performance_midi":
                    return bool(row.get("performance_midi_path", ""))
                if material == "audio":
                    return bool(row.get("audio_path", ""))
                return bool(row.get("score_path", "") or row.get("midi_path", ""))

            missing_success_sources = [
                row.get(id_field, "")
                for row in successful
                if not has_source(row)
            ]
            if missing_success_sources:
                bundle_errors.append(f"successful rows lack source paths: {', '.join(missing_success_sources[:5])}")
        failed = [row for row in rows if row.get("status") != "success"]
        total_failures += len(failed)
        silent = [row.get(id_field, "") for row in failed if not row.get("status")]
        if silent:
            bundle_errors.append(f"failed/unembedded rows lack explicit status: {', '.join(silent[:5])}")
        if failed:
            bundle_warnings.append(f"{len(failed)} rows were explicitly not embedded")
        errors.extend(f"{name}: {value}" for value in bundle_errors)
        warnings.extend(f"{name}: {value}" for value in bundle_warnings)
        bundles[name] = {
            "directory": str(directory),
            "shape": list(matrix.shape),
            "metadata_rows": len(rows),
            "successful_rows": len(successful),
            "failed_rows": len(failed),
            "model_space": metadata.get("model_space"),
            "normalization": metadata.get("normalization"),
            "embedding_sha256": actual_hash,
            "errors": bundle_errors,
            "warnings": bundle_warnings,
        }
    for space, values in dimensions.items():
        if len(values) > 1:
            errors.append(f"{space}: inconsistent embedding dimensions {sorted(values)}")
    report = {
        "schema_version": 1,
        "status": "error" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "bundles": bundles,
        "total_explicit_failures": total_failures,
        "determinism_check": "Matrix SHA-256 values match the hashes recorded at write time; rerun equality requires comparing these hashes across run manifests.",
    }
    report_dir = analysis_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "embedding_validation.json"
    text_path = report_dir / "embedding_validation.txt"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [f"Embedding validation: {report['status']}", ""]
    lines.extend(f"ERROR: {value}" for value in errors)
    lines.extend(f"WARNING: {value}" for value in warnings)
    if not errors and not warnings:
        lines.append("All configured embedding bundles passed.")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report["json_report"] = str(json_path)
    report["text_report"] = str(text_path)
    return report
