"""Study-readiness audit and manifest freezing for the CPC-based pipeline."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv

from .data_loading import (
    RECORDING_REGISTRATION_FIELDS,
    SYMBOLIC_REGISTRATION_FIELDS,
    collect_registration_conflicts,
    format_registration_conflicts,
    read_corpus_manifest,
    resolve_corpus_asset,
)
from .embedding_io import content_hash, sha256_path
from .metadata import portable_configuration, portable_path
from .passage_generation import aligned_measure_coverage, load_alignment_times


DEFAULT_AUTHORIZED_RIGHTS = (
    "public_domain",
    "licensed_for_research",
    "user_supplied_authorized",
)


class StudyAuditError(RuntimeError):
    """Raised when the corpus freeze or readiness audit cannot proceed safely."""


def _slug(value: str) -> str:
    text = "".join(character.lower() if character.isalnum() else "-" for character in value.strip())
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-") or "study-freeze"


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def build_study_readiness_report(
    rows: Iterable[Mapping[str, str]],
    *,
    corpus_root: str | Path,
    alignment_root: str | Path | None = None,
    authorized_rights: Sequence[str] = DEFAULT_AUTHORIZED_RIGHTS,
    source_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize what is safe, ambiguous, and ready before embedding runs."""

    ordered_rows = sorted(
        rows,
        key=lambda row: (
            _text(row.get("composer", "")),
            _text(row.get("composition_id", "")),
            _text(row.get("work_id", "")),
            _text(row.get("movement_id", "")),
            _text(row.get("recording_id", "")),
            _text(row.get("passage_id", "")),
        ),
    )
    corpus = Path(corpus_root)
    events = load_alignment_times(alignment_root) if alignment_root else {}
    allowed_rights = {value.strip().casefold() for value in authorized_rights}
    errors: list[str] = []
    warnings: list[str] = []

    recording_conflicts = collect_registration_conflicts(
        ordered_rows,
        key_field="recording_id",
        stable_fields=RECORDING_REGISTRATION_FIELDS,
    )
    symbolic_conflicts = collect_registration_conflicts(
        ordered_rows,
        key_field="movement_id",
        stable_fields=SYMBOLIC_REGISTRATION_FIELDS,
    )
    errors.extend(format_registration_conflicts(recording_conflicts, noun="recording_id"))
    warnings.extend(format_registration_conflicts(symbolic_conflicts, noun="movement_id"))

    composer_counts = Counter(_text(row.get("composer", "")) for row in ordered_rows)
    rights_counts = Counter(_text(row.get("rights_status", "")) for row in ordered_rows)
    audio_origin_counts = Counter(_text(row.get("audio_origin", "")) for row in ordered_rows)
    score_format_counts = Counter(_text(row.get("score_format", "")) for row in ordered_rows)
    alignment_format_counts = Counter(_text(row.get("alignment_format", "")) for row in ordered_rows)
    unique_compositions = {_text(row.get("composition_id", "")) for row in ordered_rows if _text(row.get("composition_id", ""))}
    unique_works = {_text(row.get("work_id", "")) for row in ordered_rows if _text(row.get("work_id", ""))}
    unique_movements = {_text(row.get("movement_id", "")) for row in ordered_rows if _text(row.get("movement_id", ""))}
    unique_recordings = {_text(row.get("recording_id", "")) for row in ordered_rows if _text(row.get("recording_id", ""))}

    symbolic_ready = 0
    local_audio_rows = 0
    authorized_audio_rows = 0
    unauthorized_audio_rows = 0
    fully_aligned_rows = 0
    audio_window_ready_rows = 0
    missing_symbolic_paths: list[str] = []
    missing_audio_paths: list[str] = []
    alignment_gaps: list[dict[str, Any]] = []
    audio_path_to_recordings: dict[str, set[str]] = defaultdict(set)
    symbolic_path_to_movements: dict[str, set[str]] = defaultdict(set)
    recording_by_composition: dict[str, set[str]] = defaultdict(set)
    origins_by_composition: dict[str, set[str]] = defaultdict(set)

    for row in ordered_rows:
        composition_id = _text(row.get("composition_id", ""))
        recording_id = _text(row.get("recording_id", ""))
        movement_id = _text(row.get("movement_id", ""))
        audio_origin = _text(row.get("audio_origin", ""))
        if composition_id and recording_id:
            recording_by_composition[composition_id].add(recording_id)
            if audio_origin:
                origins_by_composition[composition_id].add(audio_origin)

        symbolic_value = (
            _text(row.get("mxl_path", ""))
            or _text(row.get("score_path", ""))
            or _text(row.get("midi_path", ""))
        )
        if symbolic_value:
            symbolic_path_to_movements[symbolic_value].add(movement_id or _text(row.get("passage_id", "")))
            try:
                if resolve_corpus_asset(corpus, symbolic_value).is_file():
                    symbolic_ready += 1
                else:
                    missing_symbolic_paths.append(symbolic_value)
            except Exception:
                errors.append(f"invalid symbolic path recorded for {movement_id or row.get('passage_id', '')!r}: {symbolic_value}")

        audio_value = _text(row.get("audio_path", ""))
        if audio_value:
            audio_path_to_recordings[audio_value].add(recording_id or _text(row.get("passage_id", "")))
            try:
                if resolve_corpus_asset(corpus, audio_value).is_file():
                    local_audio_rows += 1
                    if _text(row.get("rights_status", "")).casefold() in allowed_rights:
                        authorized_audio_rows += 1
                        coverage = aligned_measure_coverage(row, events) if events else {
                            "fully_covered": False,
                            "missing_measures": list(range(int(row["bar_start"]), int(row["bar_end"]) + 1)),
                            "covered_measure_count": 0,
                            "expected_measure_count": int(row["bar_end"]) - int(row["bar_start"]) + 1,
                            "onset_seconds": None,
                            "offset_seconds": None,
                        }
                        if coverage["fully_covered"]:
                            fully_aligned_rows += 1
                            audio_window_ready_rows += 1
                        else:
                            missing = coverage["missing_measures"]
                            alignment_gaps.append(
                                {
                                    "recording_id": recording_id,
                                    "movement_id": movement_id,
                                    "bar_start": int(row["bar_start"]),
                                    "bar_end": int(row["bar_end"]),
                                    "missing_measures": missing,
                                }
                            )
                    else:
                        unauthorized_audio_rows += 1
                else:
                    missing_audio_paths.append(audio_value)
            except Exception:
                errors.append(f"invalid audio path recorded for {recording_id or row.get('passage_id', '')!r}: {audio_value}")

    repeated_audio_paths = {
        path: sorted(values)
        for path, values in sorted(audio_path_to_recordings.items())
        if path and len(values) > 1
    }
    repeated_symbolic_paths = {
        path: sorted(values)
        for path, values in sorted(symbolic_path_to_movements.items())
        if path and len(values) > 1
    }
    if repeated_audio_paths:
        warnings.append("one or more audio_path values are shared across recording_id values")
    if repeated_symbolic_paths:
        warnings.append("one or more symbolic source paths are shared across movement_id values")
    if missing_symbolic_paths:
        warnings.append(f"{len(missing_symbolic_paths)} symbolic rows reference files not present locally")
    if missing_audio_paths:
        warnings.append(f"{len(missing_audio_paths)} audio rows reference files not present locally")
    if unauthorized_audio_rows:
        warnings.append(f"{unauthorized_audio_rows} local audio rows are present but not authorized for embedding")
    if alignment_gaps:
        warnings.append(f"{len(alignment_gaps)} rows do not have full measure coverage for audio passage extraction")

    multi_interpretation = []
    for composition_id, recording_ids in sorted(recording_by_composition.items()):
        if len(recording_ids) < 2:
            continue
        multi_interpretation.append(
            {
                "composition_id": composition_id,
                "recording_count": len(recording_ids),
                "recording_ids": sorted(recording_ids),
                "audio_origins": sorted(origins_by_composition.get(composition_id, set())),
            }
        )

    cohort_recording_counts: dict[str, int] = {}
    for composer in sorted(composer_counts):
        for audio_origin in sorted(
            {
                _text(row.get("audio_origin", ""))
                for row in ordered_rows
                if _text(row.get("composer", "")) == composer
            }
        ):
            recording_ids = {
                _text(row.get("recording_id", ""))
                for row in ordered_rows
                if _text(row.get("composer", "")) == composer and _text(row.get("audio_origin", "")) == audio_origin
            }
            if audio_origin:
                cohort_recording_counts[f"{audio_origin}|{composer}"] = len({value for value in recording_ids if value})

    status = "error" if errors else "warning" if warnings else "ok"
    return {
        "schema_version": 2,
        "status": status,
        "source_manifest": str(source_manifest_path) if source_manifest_path else None,
        "source_manifest_sha256": sha256_path(source_manifest_path) if source_manifest_path and Path(source_manifest_path).is_file() else None,
        "row_count": len(ordered_rows),
        "composition_count": len(unique_compositions),
        "work_count": len(unique_works),
        "movement_count": len(unique_movements),
        "recording_count": len(unique_recordings),
        "composer_counts": dict(sorted(composer_counts.items())),
        "rights_status_counts": dict(sorted(rights_counts.items())),
        "audio_origin_counts": dict(sorted(audio_origin_counts.items())),
        "score_format_counts": dict(sorted(score_format_counts.items())),
        "alignment_format_counts": dict(sorted(alignment_format_counts.items())),
        "required_grouping_key": "composition_id",
        "required_stratification_key": "audio_origin",
        "ready_for_unstratified_composer_claim": False,
        "cohort_recording_counts": cohort_recording_counts,
        "symbolic_ready_rows": symbolic_ready,
        "local_audio_rows": local_audio_rows,
        "authorized_audio_rows": authorized_audio_rows,
        "unauthorized_audio_rows": unauthorized_audio_rows,
        "fully_aligned_audio_rows": fully_aligned_rows,
        "audio_window_ready_rows": audio_window_ready_rows,
        "multi_interpretation_compositions": multi_interpretation,
        "recording_registration_conflicts": recording_conflicts,
        "symbolic_registration_conflicts": symbolic_conflicts,
        "shared_audio_paths": repeated_audio_paths,
        "shared_symbolic_paths": repeated_symbolic_paths,
        "alignment_gaps": alignment_gaps,
        "errors": errors,
        "warnings": warnings,
        "note": "Freeze a manifest only after recording registrations, rights labels, and alignment coverage are stable.",
    }


def write_study_readiness_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(report), indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_clean_symbolic_manifest(
    rows: Iterable[Mapping[str, str]],
) -> dict[str, Any]:
    """Drop movements whose symbolic registrations are ambiguous."""

    materialized = [dict(row) for row in rows]
    conflicts = collect_registration_conflicts(
        materialized,
        key_field="movement_id",
        stable_fields=SYMBOLIC_REGISTRATION_FIELDS,
    )
    conflicting_keys = {
        _text(conflict.get("key", ""))
        for conflict in conflicts
        if _text(conflict.get("key", ""))
    }
    kept_rows = [
        row
        for row in materialized
        if _text(row.get("movement_id", "")) not in conflicting_keys
    ]
    excluded_rows = [
        row
        for row in materialized
        if _text(row.get("movement_id", "")) in conflicting_keys
    ]
    kept_movements = {
        _text(row.get("movement_id", ""))
        for row in kept_rows
        if _text(row.get("movement_id", ""))
    }
    excluded_movements = sorted(conflicting_keys)
    return {
        "schema_version": 1,
        "input_row_count": len(materialized),
        "kept_row_count": len(kept_rows),
        "excluded_row_count": len(excluded_rows),
        "kept_movement_count": len(kept_movements),
        "excluded_movement_count": len(excluded_movements),
        "excluded_movement_ids": excluded_movements,
        "symbolic_registration_conflicts": conflicts,
        "rows": kept_rows,
    }


def write_clean_symbolic_manifest_bundle(
    output_root: str | Path,
    *,
    rows: Iterable[Mapping[str, str]],
    source_manifest_path: str | Path,
    force: bool = False,
) -> dict[str, Path]:
    """Write a filtered manifest and a JSON report describing exclusions."""

    report = build_clean_symbolic_manifest(rows)
    destination = Path(output_root)
    if destination.exists() and not force:
        raise StudyAuditError(f"clean symbolic manifest destination already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    kept_rows = report.pop("rows")
    manifest_path = destination / "piano_clamp_manifest.csv"
    if not kept_rows:
        raise StudyAuditError("clean symbolic manifest would be empty after excluding conflicts")

    fieldnames: list[str] = []
    for row in kept_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    report_path = destination / "clean_symbolic_manifest_report.json"
    payload = {
        **report,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": (
            sha256_path(source_manifest_path)
            if Path(source_manifest_path).is_file()
            else None
        ),
        "clean_manifest_sha256": sha256_path(manifest_path),
    }
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"manifest_path": manifest_path, "report_path": report_path}


def build_clean_audio_render_manifest(
    render_rows: Iterable[Mapping[str, str]],
    export_rows: Iterable[Mapping[str, str]],
) -> dict[str, Any]:
    """Reconcile a render manifest against the active CPC export manifest."""

    materialized_render_rows = [dict(row) for row in render_rows]
    materialized_export_rows = [dict(row) for row in export_rows]

    render_by_recording: dict[str, dict[str, str]] = {}
    duplicate_render_recording_ids: list[str] = []
    render_output_dirs: set[str] = set()
    for row in materialized_render_rows:
        recording_id = _text(row.get("output_recording_id", ""))
        if not recording_id:
            continue
        if recording_id in render_by_recording:
            duplicate_render_recording_ids.append(recording_id)
            continue
        render_by_recording[recording_id] = row
        output_audio_path = _text(row.get("output_audio_path", ""))
        if output_audio_path:
            render_output_dirs.add(str(Path(output_audio_path).parent))

    export_by_recording: dict[str, dict[str, str]] = {}
    duplicate_export_recording_ids: list[str] = []
    for row in materialized_export_rows:
        recording_id = _text(row.get("recording_id", ""))
        if not recording_id:
            continue
        if recording_id in export_by_recording:
            duplicate_export_recording_ids.append(recording_id)
            continue
        export_by_recording[recording_id] = row

    matched_rows: list[dict[str, str]] = []
    unmatched_render_rows: list[dict[str, str]] = []
    for recording_id, render_row in sorted(render_by_recording.items()):
        export_row = export_by_recording.get(recording_id)
        if export_row is None:
            unmatched_render_rows.append(dict(render_row))
            continue
        merged = dict(export_row)
        merged["render_recording_id"] = recording_id
        merged["render_work_title"] = _text(render_row.get("work_title", ""))
        merged["render_source_dataset"] = _text(render_row.get("source_dataset", ""))
        merged["render_performance_kind"] = _text(render_row.get("performance_kind", ""))
        merged["render_status"] = _text(render_row.get("render_status", ""))
        merged["render_qc_notes"] = _text(render_row.get("qc_notes", ""))
        merged["render_output_audio_path"] = _text(render_row.get("output_audio_path", ""))
        merged["render_output_bytes"] = _text(render_row.get("output_bytes", ""))
        merged["render_output_sha256"] = _text(render_row.get("output_sha256", ""))
        merged["render_output_duration_sec"] = _text(render_row.get("output_duration_sec", ""))
        merged["render_output_sample_rate_hz"] = _text(render_row.get("output_sample_rate_hz", ""))
        merged["render_output_channels"] = _text(render_row.get("output_channels", ""))
        merged["render_peak_dbfs"] = _text(render_row.get("peak_dbfs", ""))
        merged["render_rms_dbfs"] = _text(render_row.get("rms_dbfs", ""))
        merged["render_manifest_alignment_id"] = _text(render_row.get("alignment_id", ""))
        matched_rows.append(merged)

    stale_export_rows: list[dict[str, str]] = []
    for recording_id, export_row in sorted(export_by_recording.items()):
        if recording_id in render_by_recording:
            continue
        audio_path = _text(export_row.get("audio_path", ""))
        if not audio_path:
            continue
        parent = str(Path(audio_path).parent)
        if parent not in render_output_dirs:
            continue
        stale_export_rows.append(dict(export_row))

    return {
        "schema_version": 1,
        "render_row_count": len(materialized_render_rows),
        "export_row_count": len(materialized_export_rows),
        "matched_row_count": len(matched_rows),
        "unmatched_render_row_count": len(unmatched_render_rows),
        "stale_export_row_count": len(stale_export_rows),
        "render_output_directories": sorted(render_output_dirs),
        "duplicate_render_recording_ids": sorted(set(duplicate_render_recording_ids)),
        "duplicate_export_recording_ids": sorted(set(duplicate_export_recording_ids)),
        "unmatched_render_recording_ids": [
            _text(row.get("output_recording_id", ""))
            for row in unmatched_render_rows
        ],
        "stale_export_recording_ids": [
            _text(row.get("recording_id", ""))
            for row in stale_export_rows
        ],
        "matched_rows": matched_rows,
        "unmatched_render_rows": unmatched_render_rows,
        "stale_export_rows": stale_export_rows,
    }


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_clean_audio_render_manifest_bundle(
    output_root: str | Path,
    *,
    render_rows: Iterable[Mapping[str, str]],
    export_rows: Iterable[Mapping[str, str]],
    render_manifest_path: str | Path,
    export_manifest_path: str | Path,
    force: bool = False,
) -> dict[str, Path]:
    """Write a matched render/export manifest plus reconciliation reports."""

    report = build_clean_audio_render_manifest(render_rows, export_rows)
    destination = Path(output_root)
    if destination.exists() and not force:
        raise StudyAuditError(f"clean audio render manifest destination already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    matched_rows = report.pop("matched_rows")
    unmatched_render_rows = report.pop("unmatched_render_rows")
    stale_export_rows = report.pop("stale_export_rows")
    if not matched_rows:
        raise StudyAuditError("clean audio render manifest would be empty after reconciliation")

    manifest_path = destination / "piano_clamp_manifest.csv"
    unmatched_path = destination / "unmatched_render_rows.csv"
    stale_path = destination / "stale_export_rows.csv"
    _write_csv_rows(manifest_path, matched_rows)
    _write_csv_rows(unmatched_path, unmatched_render_rows)
    _write_csv_rows(stale_path, stale_export_rows)

    report_path = destination / "clean_audio_render_manifest_report.json"
    payload = {
        **report,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "render_manifest": str(render_manifest_path),
        "render_manifest_sha256": (
            sha256_path(render_manifest_path)
            if Path(render_manifest_path).is_file()
            else None
        ),
        "export_manifest": str(export_manifest_path),
        "export_manifest_sha256": (
            sha256_path(export_manifest_path)
            if Path(export_manifest_path).is_file()
            else None
        ),
        "clean_manifest_sha256": sha256_path(manifest_path),
        "unmatched_render_rows_path": unmatched_path.name,
        "stale_export_rows_path": stale_path.name,
    }
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "manifest_path": manifest_path,
        "report_path": report_path,
        "unmatched_render_rows_path": unmatched_path,
        "stale_export_rows_path": stale_path,
    }


def freeze_study_manifest(
    config: Mapping[str, Any],
    *,
    label: str,
    force: bool = False,
    allow_errors: bool = False,
) -> Path:
    """Copy the current manifest plus audit metadata into a versioned freeze directory."""

    manifest_path = Path(str(config["corpus_manifest"]))
    rows = read_corpus_manifest(manifest_path)
    report = build_study_readiness_report(
        rows,
        corpus_root=str(config["corpus_root"]),
        alignment_root=str(config.get("alignment_root", "")) or None,
        authorized_rights=tuple(config.get("authorized_rights_statuses", DEFAULT_AUTHORIZED_RIGHTS)),
        source_manifest_path=manifest_path,
    )
    if report["status"] == "error" and not allow_errors:
        raise StudyAuditError("study readiness report contains blocking errors; pass --allow-errors to freeze anyway")

    analysis_root = Path(str(config["analysis_root"]))
    destination = analysis_root / "frozen_manifests" / _slug(label)
    if destination.exists() and not force:
        raise StudyAuditError(f"freeze destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".piano-clamp-freeze-", dir=destination.parent) as temporary:
        stage = Path(temporary) / destination.name
        stage.mkdir()
        frozen_manifest = stage / "piano_clamp_manifest.csv"
        frozen_manifest.write_bytes(manifest_path.read_bytes())
        readiness_path = write_study_readiness_report(stage / "study_readiness.json", report)
        metadata = {
            "schema_version": 1,
            "label": label,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_manifest": portable_path(manifest_path, config),
            "source_manifest_sha256": sha256_path(manifest_path),
            "frozen_manifest_sha256": sha256_path(frozen_manifest),
            "study_readiness_path": readiness_path.name,
            "study_readiness_status": report["status"],
            "configuration_hash": content_hash(portable_configuration(config)),
        }
        (stage / "freeze_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(destination)
        stage.replace(destination)
    return destination
