"""Build a native Piano CLaMP adapter from canonical CPC tables.

The classical-performance-corpus remains read-only. This module validates its
canonical IDs and local assets, derives a consumer-specific study subset from
the corpus's canonical tables, filters the global alignment stores, derives
measure-level performance times, and writes a small adapter directory consumed
directly by Piano CLaMP's existing embedding and passage commands.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
import unicodedata
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_loading import CORPUS_MANIFEST_FIELDS, read_corpus_manifest
from .study import DEFAULT_AUTHORIZED_RIGHTS, build_study_readiness_report


CPC_SCHEMA_VERSION = "1.1.0"
DEFAULT_COMPOSERS = ("Chopin", "Mozart")
SUPPORTED_SCORE_EXTENSIONS = {".mid", ".midi", ".mxl", ".musicxml", ".xml"}
ADAPTER_EXTRA_FIELDS = (
    "composition_id",
    "work_title",
    "score_version_id",
    "performance_id",
    "alignment_id",
    "performance_midi_path",
    "audio_origin",
    "alignment_granularity",
    "alignment_method",
    "alignment_quality",
    "alignment_confidence",
    "source_record_ids",
    "source_dataset",
    "licenses",
)
ALIGNMENT_EVENT_FIELDS = (
    "recording_id",
    "movement_id",
    "alignment_id",
    "score_measure",
    "score_measure_label",
    "onset_seconds",
    "offset_seconds",
    "confidence",
    "granularity",
    "method",
    "quality_label",
)
CPC_REQUIRED_COLUMNS = (
    "export_row_id",
    "corpus_release",
    "composer",
    "composition_id",
    "work_id",
    "score_version_id",
    "performance_id",
    "recording_id",
    "alignment_id",
    "score_path",
    "audio_path",
    "alignment_path",
    "audio_available",
    "score_available",
    "alignment_available",
    "alignment_quality",
    "alignment_confidence",
    "eligible_for_clap",
    "synthetic_audio",
    "source_record_ids",
    "licenses",
)
ID_PREFIXES = {
    "composition_id": "cmp_",
    "work_id": "wrk_",
    "score_version_id": "scv_",
    "performance_id": "prf_",
    "recording_id": "rec_",
    "alignment_id": "aln_",
}
COMPAT_EXPORT_DEFAULT = "manifests/exports/piano-clamp-chopin-mozart.csv"


class CpcAdapterError(ValueError):
    """Raised when the CPC export cannot safely feed Piano CLaMP."""


@dataclass(frozen=True)
class CpcAdapterSummary:
    corpus_root: str
    export_manifest: str
    corpus_release: str
    schema_version: str
    target_rows: int
    imported_rows: int
    timing_rows: int
    excluded_rows: int
    score_manifest: str
    alignment_events: str
    issues_manifest: str
    cohort_summary: str
    readiness_report: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CpcAdapterSupplementSummary:
    source_adapter_root: str
    output_root: str
    render_manifest: str
    added_rows: int
    total_rows: int
    added_timing_rows: int
    total_timing_rows: int
    unmatched_render_rows: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    result = str(value).strip()
    return "" if result.casefold() == "nan" else result


def _truthy(value: object) -> bool:
    return _text(value).casefold() in {"1", "true", "yes", "y"}


def _composer(value: object) -> str:
    key = unicodedata.normalize("NFKD", _text(value)).encode("ascii", "ignore").decode().casefold()
    if "chopin" in key:
        return "Frédéric Chopin"
    if "mozart" in key:
        return "Wolfgang Amadeus Mozart"
    if "bach" in key:
        return "Johann Sebastian Bach"
    return _text(value)


def _composer_key(value: object) -> str:
    normalized = _composer(value)
    if "Chopin" in normalized:
        return "Chopin"
    if "Mozart" in normalized:
        return "Mozart"
    if "Bach" in normalized:
        return "Bach"
    return normalized


def _period(composer: str) -> str:
    if "Chopin" in composer:
        return "Romantic"
    if "Mozart" in composer:
        return "Classical"
    if "Bach" in composer:
        return "Baroque"
    return ""


def _validate_id(field: str, value: object) -> str:
    identifier = _text(value)
    prefix = ID_PREFIXES[field]
    if not identifier.startswith(prefix) or len(identifier) <= len(prefix):
        raise CpcAdapterError(f"{field} must be an opaque ID beginning with {prefix}")
    return identifier


def _resolve_asset(corpus_root: Path, value: object) -> Path:
    relative = Path(_text(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise CpcAdapterError(f"CPC asset path must remain repository-relative: {value}")
    resolved = (corpus_root / relative).resolve(strict=False)
    if resolved != corpus_root and corpus_root not in resolved.parents:
        raise CpcAdapterError(f"CPC asset path escapes the corpus root: {value}")
    return resolved


def _sha256(path: Path, cache: dict[Path, str]) -> str:
    resolved = path.resolve()
    if resolved in cache:
        return cache[resolved]
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result = digest.hexdigest()
    cache[resolved] = result
    return result


def _validate_hash(
    expected_value: object,
    *,
    field: str,
    path: Path,
    cache: dict[Path, str],
    verify: bool,
) -> str:
    expected = _text(expected_value).casefold()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise CpcAdapterError(f"{field} is not a valid SHA-256 digest")
    if verify:
        actual = _sha256(path, cache)
        if actual != expected:
            raise CpcAdapterError(f"{field} does not match {path}: expected {expected}, found {actual}")
    return expected


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise CpcAdapterError(f"CSV file does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _unique_table(path: Path, id_column: str) -> pd.DataFrame:
    if not path.is_file():
        raise CpcAdapterError(f"canonical table does not exist: {path}")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if id_column not in frame:
        raise CpcAdapterError(f"canonical table {path} is missing {id_column}")
    if frame[id_column].duplicated().any():
        raise CpcAdapterError(f"canonical table {path} contains duplicate {id_column} values")
    return frame.set_index(id_column, drop=False)


def _release_from_manifests(corpus_root: Path) -> str:
    releases = sorted((corpus_root / "manifests" / "releases").glob("*.json"))
    if not releases:
        return "unknown"
    for path in reversed(releases):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        release = _text(payload.get("release")) or _text(payload.get("corpus_release"))
        if release:
            return release
    return releases[-1].stem


def _selected_alignment_rows(path: Path, alignment_ids: set[str]) -> dict[str, pd.DataFrame]:
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, chunksize=250_000):
        if "alignment_id" not in chunk:
            raise CpcAdapterError(f"canonical alignment store is missing alignment_id: {path}")
        retained = chunk[chunk["alignment_id"].astype(str).isin(alignment_ids)]
        if not retained.empty:
            selected.append(retained)
    if not selected:
        return {}
    combined = pd.concat(selected, ignore_index=True)
    return {str(key): group.copy() for key, group in combined.groupby("alignment_id", sort=False)}


def _wav_duration(path: Path) -> float | None:
    if path.suffix.casefold() != ".wav":
        return None
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            return handle.getnframes() / rate if rate > 0 else None
    except (OSError, EOFError, wave.Error):
        return None


def _event_timings(events: pd.DataFrame, *, audio_duration: float | None) -> list[dict[str, object]]:
    required = {"measure_index", "performance_time_sec", "confidence"}
    if missing := sorted(required - set(events.columns)):
        raise CpcAdapterError(f"beat alignment is missing: {', '.join(missing)}")
    frame = events.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
    frame = frame[(frame["measure_index"] >= 1) & (frame["performance_time_sec"] >= 0)]
    if frame.empty:
        raise CpcAdapterError("beat alignment contains no usable anchors")
    grouped = frame.groupby("measure_index", sort=True)
    starts = grouped["performance_time_sec"].min()
    confidence = grouped["confidence"].min()
    measures = [int(value) for value in starts.index]
    if measures != list(range(1, max(measures) + 1)):
        raise CpcAdapterError("beat alignment measures are not contiguous and one-based")
    values = starts.to_numpy(float)
    anchors = np.sort(frame["performance_time_sec"].unique().astype(float))
    steps = np.diff(anchors)
    steps = steps[steps > 0]
    fallback = float(np.median(steps)) if len(steps) else 0.5
    final_end = max(float(frame["performance_time_sec"].max()) + fallback, values[-1] + 1e-3)
    if audio_duration is not None:
        final_end = min(final_end, audio_duration)
    rows: list[dict[str, object]] = []
    for index, measure in enumerate(measures):
        onset = float(values[index])
        offset = float(values[index + 1]) if index + 1 < len(values) else final_end
        if offset <= onset:
            raise CpcAdapterError(f"beat alignment has a non-positive interval at measure {measure}")
        rows.append(
            {
                "score_measure": measure,
                "score_measure_label": str(measure),
                "onset_seconds": onset,
                "offset_seconds": offset,
                "confidence": float(confidence.loc[measure]),
            }
        )
    return rows


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _score_measure_times(score_path: Path) -> list[dict[str, object]]:
    try:
        from music21 import converter, stream, tempo
    except ImportError as exc:
        raise CpcAdapterError(
            "music21 is required to convert CPC note alignments and crop MIDI passages"
        ) from exc
    try:
        score = converter.parse(str(score_path))
    except Exception as exc:
        raise CpcAdapterError(f"could not parse score timing from {score_path}: {exc}") from exc
    parts = list(getattr(score, "parts", [])) or [score]
    measures: list[Any] = []
    for part in parts:
        measures = list(part.getElementsByClass(stream.Measure))
        if measures:
            break
    if not measures:
        measures = list(score.recurse().getElementsByClass(stream.Measure))
    if not measures:
        raise CpcAdapterError(f"score contains no measures: {score_path}")

    tempo_events: list[tuple[float, float]] = []
    for marker in score.recurse().getElementsByClass(tempo.MetronomeMark):
        bpm = getattr(marker, "number", None)
        if bpm is None:
            continue
        try:
            offset = float(marker.getOffsetInHierarchy(score))
        except Exception:
            offset = _safe_float(getattr(marker, "offset", 0.0))
        tempo_events.append((max(0.0, offset), float(bpm)))
    tempo_events.sort()
    if not tempo_events:
        tempo_events = [(0.0, 120.0)]
    elif tempo_events[0][0] > 0:
        tempo_events.insert(0, (0.0, tempo_events[0][1]))
    deduplicated: list[tuple[float, float]] = []
    for offset, bpm in tempo_events:
        if deduplicated and offset == deduplicated[-1][0]:
            deduplicated[-1] = (offset, bpm)
        else:
            deduplicated.append((offset, bpm))

    def seconds(quarter_offset: float) -> float:
        total = 0.0
        current_offset = 0.0
        current_bpm = deduplicated[0][1]
        for event_offset, event_bpm in deduplicated[1:]:
            if quarter_offset <= event_offset:
                break
            total += max(0.0, event_offset - current_offset) * 60.0 / current_bpm
            current_offset = event_offset
            current_bpm = event_bpm
        return total + max(0.0, quarter_offset - current_offset) * 60.0 / current_bpm

    output: list[dict[str, object]] = []
    for index, measure in enumerate(measures, start=1):
        start_quarter = _safe_float(getattr(measure, "offset", 0.0))
        if index < len(measures):
            end_quarter = _safe_float(getattr(measures[index], "offset", start_quarter))
        else:
            duration = _safe_float(getattr(getattr(measure, "duration", None), "quarterLength", 0.0))
            end_quarter = start_quarter + duration
        label = getattr(measure, "number", index)
        output.append(
            {
                "score_measure_label": str(label),
                "score_start": seconds(start_quarter),
                "score_end": seconds(end_quarter),
            }
        )
    return output


def _interpolator(score_times: np.ndarray, performance_times: np.ndarray):
    order = np.argsort(score_times, kind="stable")
    pairs = pd.DataFrame({"score": score_times[order], "performance": performance_times[order]})
    pairs = pairs.groupby("score", as_index=False)["performance"].median().sort_values("score")
    x = pairs["score"].to_numpy(float)
    y = np.maximum.accumulate(pairs["performance"].to_numpy(float))
    if len(x) < 2 or x[-1] <= x[0]:
        raise CpcAdapterError("note alignment needs at least two score-time anchors")

    def interpolate(values: np.ndarray) -> np.ndarray:
        result = np.interp(values, x, y)
        left_slope = max(0.0, (y[1] - y[0]) / (x[1] - x[0]))
        right_slope = max(0.0, (y[-1] - y[-2]) / (x[-1] - x[-2]))
        left = values < x[0]
        right = values > x[-1]
        result[left] = y[0] + (values[left] - x[0]) * left_slope
        result[right] = y[-1] + (values[right] - x[-1]) * right_slope
        return result

    return interpolate


def _note_timings(
    notes: pd.DataFrame,
    *,
    score_path: Path,
    audio_duration: float | None,
) -> list[dict[str, object]]:
    required = {"score_onset_sec", "performance_onset_sec", "confidence"}
    if missing := sorted(required - set(notes.columns)):
        raise CpcAdapterError(f"note alignment is missing: {', '.join(missing)}")
    frame = notes.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
    if "mapping_state" in frame:
        frame = frame[frame["mapping_state"].astype(str).str.casefold().isin({"matched", "interpolated"})]
    if frame.empty:
        raise CpcAdapterError("note alignment contains no usable anchors")
    measures = _score_measure_times(score_path)
    interpolate = _interpolator(
        frame["score_onset_sec"].to_numpy(float),
        frame["performance_onset_sec"].to_numpy(float),
    )
    starts = np.maximum(interpolate(np.asarray([row["score_start"] for row in measures], dtype=float)), 0.0)
    ends = interpolate(np.asarray([row["score_end"] for row in measures], dtype=float))
    if audio_duration is not None:
        starts = np.minimum(starts, audio_duration)
        ends = np.minimum(ends, audio_duration)
    confidence = float(frame["confidence"].min())
    output: list[dict[str, object]] = []
    for index, (measure, onset, offset) in enumerate(zip(measures, starts, ends), start=1):
        if offset <= onset:
            continue
        output.append(
            {
                "score_measure": index,
                "score_measure_label": measure["score_measure_label"],
                "onset_seconds": float(onset),
                "offset_seconds": float(offset),
                "confidence": confidence,
            }
        )
    if not output:
        raise CpcAdapterError("note alignment interpolation produced no positive measure intervals")
    return output


def _rights_status(licenses: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", licenses.casefold()).strip()
    if "cc by" in normalized or "mit" in normalized or "public domain" in normalized:
        return "licensed_for_research"
    return "license_review_required"


def _supported_score_path(score_path: str) -> bool:
    return Path(score_path).suffix.casefold() in SUPPORTED_SCORE_EXTENSIONS


def _pick_render_export_candidate(
    rows: Iterable[dict[str, str]],
    *,
    performance_id: str,
    work_id: str,
    score_version_id: str,
    alignment_id: str,
) -> dict[str, str] | None:
    candidates = [
        dict(row)
        for row in rows
        if _text(row.get("performance_id")) == performance_id
        and _text(row.get("work_id")) == work_id
    ]
    if not candidates:
        return None

    for key, value in (
        ("score_version_id", score_version_id),
        ("alignment_id", alignment_id),
    ):
        if value:
            narrowed = [row for row in candidates if _text(row.get(key)) == value]
            if narrowed:
                candidates = narrowed

    for predicate in (
        lambda row: _truthy(row.get("synthetic_audio")),
        lambda row: _truthy(row.get("eligible_for_clap")),
    ):
        narrowed = [row for row in candidates if predicate(row)]
        if narrowed:
            candidates = narrowed

    if len(candidates) != 1:
        return None
    return candidates[0]


def _select_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    selected = frame.copy()
    for column in columns:
        if column not in selected.columns:
            selected[column] = ""
    return selected.loc[:, list(columns)]


def _canonical_source_manifest(
    *,
    corpus: Path,
    requested: set[str] | None,
    works: pd.DataFrame,
    compositions: pd.DataFrame,
    scores: pd.DataFrame,
    performances: pd.DataFrame,
    recordings: pd.DataFrame,
    alignments: pd.DataFrame,
    source_records: pd.DataFrame,
) -> pd.DataFrame:
    audio_recordings = recordings[
        (recordings["media_type"].map(_text).str.casefold() == "audio")
        & (recordings["qc_status"].map(_text).str.casefold() == "pass")
    ].copy()
    if audio_recordings.empty:
        return pd.DataFrame(columns=CPC_REQUIRED_COLUMNS)
    midi_lookup = _performance_midi_lookup(recordings)

    export = (
        audio_recordings
        .merge(
            performances[["performance_id", "work_id", "score_version_id"]],
            on="performance_id",
            how="inner",
        )
        .merge(
            _select_columns(
                alignments,
                (
                    "alignment_id",
                    "work_id",
                    "score_version_id",
                    "performance_id",
                    "recording_id",
                    "granularity",
                    "method",
                    "quality_label",
                    "confidence",
                    "qc_status",
                    "source_record_ids",
                    "source_file_path",
                ),
            ),
            on=["performance_id", "recording_id"],
            how="inner",
            suffixes=("", "_alignment"),
        )
        .merge(
            _select_columns(works, ("work_id", "composition_id", "title")),
            on="work_id",
            how="inner",
        )
        .merge(
            _select_columns(compositions, ("composition_id", "composer")),
            on="composition_id",
            how="inner",
        )
        .merge(
            _select_columns(scores, ("score_version_id", "source_file_path", "sha256")).rename(
                columns={
                    "source_file_path": "score_path",
                    "sha256": "score_sha256",
                }
            ),
            on="score_version_id",
            how="inner",
        )
    )
    if export.empty:
        return pd.DataFrame(columns=CPC_REQUIRED_COLUMNS)

    if requested is not None:
        export = export[export["composer"].map(_composer_key).isin(requested)].copy()
    if export.empty:
        return pd.DataFrame(columns=CPC_REQUIRED_COLUMNS)

    def gather_source_ids(row: pd.Series) -> str:
        values: set[str] = set()
        for field in ("source_record_ids", "source_record_ids_alignment"):
            values.update(item for item in _text(row.get(field)).split(";") if item)
        for identifier in (
            _text(row.get("source_record_id")),
            _text(row.get("source_record_id_alignment")),
        ):
            if identifier:
                values.add(identifier)
        return ";".join(sorted(values))

    def licenses_for(source_ids: str) -> str:
        licenses: set[str] = set()
        for source_id in (item for item in source_ids.split(";") if item):
            if source_id not in source_records.index:
                continue
            license_text = _text(source_records.loc[source_id].get("license"))
            if license_text:
                licenses.add(license_text)
        return ";".join(sorted(licenses))

    export["source_record_ids_combined"] = export.apply(gather_source_ids, axis=1)
    export["licenses"] = export["source_record_ids_combined"].map(licenses_for)
    export["score_available"] = export["score_path"].map(
        lambda value: str(_resolve_asset(corpus, value).is_file()).lower() if _text(value) else "false"
    )
    export["audio_available"] = export["file_path"].map(
        lambda value: str(_resolve_asset(corpus, value).is_file()).lower() if _text(value) else "false"
    )
    export["alignment_available"] = export["granularity"].map(_text).str.casefold().isin({"beat", "note"})
    export["alignment_available"] = export["alignment_available"].map(lambda value: str(bool(value)).lower())
    export["eligible_for_clap"] = (
        export["score_path"].map(_supported_score_path)
        & export["score_available"].map(_truthy)
        & export["audio_available"].map(_truthy)
        & export["alignment_available"].map(_truthy)
        & export["qc_status_alignment"].map(_text).str.casefold().isin({"pass", "partial_alignment"})
    ).map(lambda value: str(bool(value)).lower())
    export["synthetic_audio"] = export["rendering_kind"].map(_text).str.casefold().eq("synthetic_audio")
    export["synthetic_audio"] = export["synthetic_audio"].map(lambda value: str(bool(value)).lower())
    export["alignment_path"] = export["granularity"].map(_text).str.casefold().map(
        lambda value: "data/canonical/alignment_notes.csv" if value == "note" else "data/canonical/alignment_events.csv"
    )
    export["performance_midi_path"] = export["performance_id"].map(lambda value: midi_lookup.get(_text(value), ""))
    export["export_row_id"] = export.apply(
        lambda row: hashlib.sha1(
            "\x1f".join(
                (
                    _text(row.get("composition_id")),
                    _text(row.get("work_id")),
                    _text(row.get("score_version_id")),
                    _text(row.get("performance_id")),
                    _text(row.get("recording_id")),
                    _text(row.get("alignment_id")),
                )
            ).encode("utf-8")
        ).hexdigest()[:12],
        axis=1,
    )
    export["export_row_id"] = export["export_row_id"].map(lambda value: f"exp_{value}")
    export["corpus_release"] = _release_from_manifests(corpus)
    export["original_source_dataset"] = export["source_dataset"]
    export["alignment_quality"] = export["quality_label"]
    export["alignment_confidence"] = export["confidence"]
    export = export.rename(
        columns={
            "file_path": "audio_path",
            "source_record_ids_combined": "source_record_ids",
        }
    )
    columns = list(CPC_REQUIRED_COLUMNS) + ["original_source_dataset", "performance_midi_path"]
    for column in columns:
        if column not in export.columns:
            export[column] = ""
    return export[columns].copy()


def _performance_midi_lookup(recordings: pd.DataFrame) -> dict[str, str]:
    if recordings.empty or "performance_id" not in recordings.columns:
        return {}
    frame = _select_columns(
        recordings,
        ("performance_id", "media_type", "rendering_kind", "file_path", "qc_status"),
    )
    frame = frame[
        (frame["media_type"].map(_text).str.casefold() == "midi")
        & (frame["qc_status"].map(_text).str.casefold() == "pass")
        & frame["file_path"].map(_text).astype(bool)
    ].copy()
    if frame.empty:
        return {}
    priority = {
        "refined_performance_midi": 0,
        "performance_midi": 1,
    }
    frame["priority"] = frame["rendering_kind"].map(lambda value: priority.get(_text(value).casefold(), 9))
    frame = frame.sort_values(["performance_id", "priority", "file_path"], kind="stable")
    return {
        _text(performance_id): _text(group.iloc[0].get("file_path"))
        for performance_id, group in frame.groupby("performance_id", sort=False)
        if _text(performance_id)
    }


def _with_performance_midi_paths(export: pd.DataFrame, recordings: pd.DataFrame) -> pd.DataFrame:
    output = export.copy()
    if "performance_midi_path" not in output.columns:
        output["performance_midi_path"] = ""
    lookup = _performance_midi_lookup(recordings)
    if not lookup:
        return output
    output["performance_midi_path"] = output.apply(
        lambda row: _text(row.get("performance_midi_path")) or lookup.get(_text(row.get("performance_id")), ""),
        axis=1,
    )
    return output


def _load_source_manifest(
    *,
    corpus: Path,
    export_manifest: str | Path | None,
    requested: set[str] | None,
) -> tuple[pd.DataFrame, str, str]:
    canonical = corpus / "data" / "canonical"
    works = _unique_table(canonical / "works.csv", "work_id")
    compositions = _unique_table(canonical / "compositions.csv", "composition_id")
    scores = _unique_table(canonical / "score_versions.csv", "score_version_id")
    performances = _unique_table(canonical / "performances.csv", "performance_id")
    recordings = _unique_table(canonical / "recordings.csv", "recording_id")
    alignments = _unique_table(canonical / "alignments.csv", "alignment_id")
    source_records = _unique_table(corpus / "metadata" / "source_records.csv", "source_record_id")

    manifest_path: Path | None = None
    if export_manifest:
        candidate = Path(export_manifest).expanduser()
        if not candidate.is_absolute():
            candidate = corpus / candidate
        candidate = candidate.resolve()
        if candidate.is_file():
            manifest_path = candidate

    if manifest_path is not None:
        export = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
        if missing := [column for column in CPC_REQUIRED_COLUMNS if column not in export.columns]:
            raise CpcAdapterError(f"CPC export is missing columns: {', '.join(missing)}")
        releases = {_text(value) for value in export["corpus_release"]}
        if len(releases) != 1 or not next(iter(releases), ""):
            raise CpcAdapterError("CPC export must contain one non-empty corpus_release")
        release = next(iter(releases))
        schema_path = manifest_path.with_name("piano-clamp-schema.json")
        if schema_path.is_file():
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            if _text(schema.get("schema_version")) != CPC_SCHEMA_VERSION:
                raise CpcAdapterError(f"expected CPC schema {CPC_SCHEMA_VERSION}: {schema_path}")
        return _with_performance_midi_paths(export, recordings), release, str(manifest_path)

    export = _canonical_source_manifest(
        corpus=corpus,
        requested=requested,
        works=works.reset_index(drop=True),
        compositions=compositions.reset_index(drop=True),
        scores=scores.reset_index(drop=True),
        performances=performances.reset_index(drop=True),
        recordings=recordings.reset_index(drop=True),
        alignments=alignments.reset_index(drop=True),
        source_records=source_records,
    )
    return export, _release_from_manifests(corpus), "canonical tables"


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_cpc_adapter(
    *,
    corpus_root: str | Path,
    output_root: str | Path,
    export_manifest: str | Path | None = None,
    composers: Iterable[str] | None = DEFAULT_COMPOSERS,
    verify_hashes: bool = True,
) -> CpcAdapterSummary:
    """Build an atomic, read-only adapter for the canonical CPC release."""

    corpus = Path(corpus_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise CpcAdapterError(f"output already exists; choose a new versioned path: {output}")
    if output == corpus or corpus in output.parents:
        raise CpcAdapterError("adapter output must stay outside the read-only corpus root")
    requested = None if composers is None else {_composer_key(value) for value in composers}
    export, release, manifest_source = _load_source_manifest(
        corpus=corpus,
        export_manifest=export_manifest,
        requested=requested,
    )
    canonical = corpus / "data" / "canonical"
    works = _unique_table(canonical / "works.csv", "work_id")
    scores = _unique_table(canonical / "score_versions.csv", "score_version_id")
    recordings = _unique_table(canonical / "recordings.csv", "recording_id")
    alignments = _unique_table(canonical / "alignments.csv", "alignment_id")
    eligible = export["eligible_for_clap"].map(_truthy)
    if requested is not None:
        eligible = eligible & export["composer"].map(_composer_key).isin(requested)
    candidate = export[
        eligible
        & export["score_path"].map(lambda value: Path(value).suffix.casefold() in SUPPORTED_SCORE_EXTENSIONS)
    ].copy()
    alignment_groups: dict[str, dict[str, pd.DataFrame]] = {}
    for relative, group in candidate.groupby("alignment_path"):
        path = _resolve_asset(corpus, relative)
        alignment_groups[_text(relative)] = _selected_alignment_rows(
            path, set(group["alignment_id"].map(_text))
        )

    hash_cache: dict[Path, str] = {}
    manifest_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    errors: list[str] = []
    target_rows = 0
    for row_index, row in export.iterrows():
        if requested is not None and _composer_key(row.get("composer")) not in requested:
            continue
        target_rows += 1
        if not _truthy(row.get("eligible_for_clap")):
            issues.append(
                {
                    "export_row": row_index + 2,
                    "recording_id": _text(row.get("recording_id")),
                    "reason": "row is not marked eligible_for_clap",
                }
            )
            continue
        score_relative = _text(row.get("score_path"))
        if Path(score_relative).suffix.casefold() not in SUPPORTED_SCORE_EXTENSIONS:
            issues.append(
                {
                    "export_row": row_index + 2,
                    "recording_id": _text(row.get("recording_id")),
                    "reason": "score_path is a metadata locator rather than MusicXML/MXL/MIDI",
                }
            )
            continue
        recording_hint = _text(row.get("recording_id"))
        try:
            ids = {field: _validate_id(field, row.get(field)) for field in ID_PREFIXES}
            work = works.loc[ids["work_id"]]
            score_ref = scores.loc[ids["score_version_id"]]
            recording_ref = recordings.loc[ids["recording_id"]]
            alignment_ref = alignments.loc[ids["alignment_id"]]
            score_path = _resolve_asset(corpus, score_relative)
            audio_path = _resolve_asset(corpus, row.get("audio_path"))
            alignment_path = _resolve_asset(corpus, row.get("alignment_path"))
            for kind, path in (("score", score_path), ("audio", audio_path), ("alignment", alignment_path)):
                if not path.is_file():
                    raise CpcAdapterError(f"{kind} asset does not exist: {path}")
            _validate_hash(
                score_ref.get("sha256"), field="score_sha256", path=score_path,
                cache=hash_cache, verify=verify_hashes,
            )
            _validate_hash(
                recording_ref.get("sha256"), field="audio_sha256", path=audio_path,
                cache=hash_cache, verify=verify_hashes,
            )
            source_alignment_hash = _text(alignment_ref.get("sha256"))
            if len(source_alignment_hash) != 64:
                raise CpcAdapterError("canonical alignment source hash is missing")
            granularity = _text(alignment_ref.get("granularity")).casefold()
            groups = alignment_groups.get(_text(row.get("alignment_path")), {})
            anchors = groups.get(ids["alignment_id"])
            if anchors is None or anchors.empty:
                raise CpcAdapterError("global alignment store has no rows for alignment_id")
            duration = _wav_duration(audio_path)
            if granularity == "note":
                timings = _note_timings(anchors, score_path=score_path, audio_duration=duration)
            else:
                timings = _event_timings(anchors, audio_duration=duration)
            composer = _composer(row.get("composer"))
            licenses = _text(row.get("licenses"))
            original_source = _text(row.get("original_source_dataset")) or _text(recording_ref.get("source_dataset"))
            alignment_quality = _text(row.get("alignment_quality"))
            audio_origin = "synthetic_rendering" if _truthy(row.get("synthetic_audio")) else "source_recording"
            fully_aligned = "false" if "partial" in alignment_quality.casefold() else "true"
            score_suffix = score_path.suffix.casefold()
            performance_midi_relative = _text(row.get("performance_midi_path"))
            manifest_rows.append(
                {
                    "passage_id": _text(row.get("export_row_id")),
                    "composer": composer,
                    "period": _period(composer),
                    "work_id": ids["work_id"],
                    "movement_id": ids["work_id"],
                    "recording_id": ids["recording_id"],
                    "score_path": score_relative,
                    "mxl_path": score_relative if score_suffix in {".mxl", ".musicxml", ".xml"} else "",
                    "audio_path": _text(row.get("audio_path")),
                    "midi_path": score_relative if score_suffix in {".mid", ".midi"} else "",
                    "performance_midi_path": performance_midi_relative,
                    "alignment_path": _text(row.get("alignment_path")),
                    "annotation_path": "",
                    "bar_start": 1,
                    "bar_end": len(timings),
                    "score_format": score_suffix.lstrip("."),
                    "alignment_format": f"cpc_canonical_{granularity}",
                    "rights_status": _rights_status(licenses),
                    "segment_type": "complete_movement",
                    "annotation_type": "",
                    "fully_aligned": fully_aligned,
                    "eligible_for_clamp": "true",
                    "availability_status": "ready",
                    "composition_id": ids["composition_id"],
                    "work_title": _text(work.get("title")) or ids["work_id"],
                    "score_version_id": ids["score_version_id"],
                    "performance_id": ids["performance_id"],
                    "alignment_id": ids["alignment_id"],
                    "audio_origin": audio_origin,
                    "alignment_granularity": granularity,
                    "alignment_method": _text(alignment_ref.get("method")),
                    "alignment_quality": alignment_quality,
                    "alignment_confidence": _text(row.get("alignment_confidence")),
                    "source_record_ids": _text(row.get("source_record_ids")),
                    "source_dataset": original_source,
                    "licenses": licenses,
                }
            )
            for timing in timings:
                timing_rows.append(
                    {
                        "recording_id": ids["recording_id"],
                        "movement_id": ids["work_id"],
                        "alignment_id": ids["alignment_id"],
                        **timing,
                        "granularity": granularity,
                        "method": _text(alignment_ref.get("method")),
                        "quality_label": alignment_quality,
                    }
                )
        except (CpcAdapterError, KeyError, OSError, TypeError, ValueError) as exc:
            errors.append(f"row {row_index + 2} ({recording_hint or 'unknown'}): {exc}")
    if errors:
        raise CpcAdapterError("eligible CPC rows failed validation:\n- " + "\n- ".join(errors))
    if not manifest_rows:
        raise CpcAdapterError("no usable CPC rows remain for the requested composers")

    manifest_rows.sort(key=lambda item: str(item["passage_id"]))
    timing_rows.sort(key=lambda item: (str(item["recording_id"]), int(item["score_measure"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".piano-clamp-cpc-", dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()
        manifest_output = stage / "piano_clamp_manifest.csv"
        timing_output = stage / "alignment_events.csv"
        issues_output = stage / "import_issues.csv"
        cohort_output = stage / "cohort_summary.csv"
        readiness_output = stage / "study_readiness.json"
        _write_csv(manifest_output, CORPUS_MANIFEST_FIELDS + ADAPTER_EXTRA_FIELDS, manifest_rows)
        _write_csv(timing_output, ALIGNMENT_EVENT_FIELDS, timing_rows)
        _write_csv(issues_output, ("export_row", "recording_id", "reason"), issues)
        cohort = (
            pd.DataFrame(manifest_rows)
            .groupby(["composer", "audio_origin", "alignment_granularity"])
            .agg(
                recording_count=("recording_id", "nunique"),
                composition_count=("composition_id", "nunique"),
                work_count=("work_id", "nunique"),
            )
            .reset_index()
        )
        cohort.to_csv(cohort_output, index=False)
        counts = (
            pd.DataFrame(manifest_rows)
            .groupby(["audio_origin", "composer"])["recording_id"]
            .nunique()
            .to_dict()
        )
        readiness = build_study_readiness_report(
            manifest_rows,
            corpus_root=corpus,
            alignment_root=stage,
            authorized_rights=DEFAULT_AUTHORIZED_RIGHTS,
            source_manifest_path=manifest_output,
        )
        readiness["cohort_recording_counts"] = {
            f"{origin}|{composer}": count for (origin, composer), count in counts.items()
        }
        readiness["note"] = "Never mix source recordings and synthetic renderings in a composer contrast."
        readiness_output.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Exercise the same schema reader used by all downstream commands.
        if len(read_corpus_manifest(manifest_output)) != len(manifest_rows):
            raise CpcAdapterError("generated adapter manifest failed its row-count invariant")
        summary = CpcAdapterSummary(
            corpus_root=str(corpus),
            export_manifest=str(manifest_source),
            corpus_release=release,
            schema_version=CPC_SCHEMA_VERSION,
            target_rows=target_rows,
            imported_rows=len(manifest_rows),
            timing_rows=len(timing_rows),
            excluded_rows=len(issues),
            score_manifest=str(output / manifest_output.name),
            alignment_events=str(output / timing_output.name),
            issues_manifest=str(output / issues_output.name),
            cohort_summary=str(output / cohort_output.name),
            readiness_report=str(output / readiness_output.name),
        )
        metadata = summary.to_dict()
        metadata.update(
            {
                "adapter_schema": "piano-clamp-cpc-adapter-1.0.0",
                "source_manifest_sha256": (
                    _sha256(Path(manifest_source), hash_cache)
                    if manifest_source not in {"canonical tables", ""}
                    else ""
                ),
                "hash_verification_performed": bool(verify_hashes),
                "source_corpus_is_read_only": True,
                "mxl_clap_dependency": False,
            }
        )
        (stage / "adapter_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stage.replace(output)
    return summary


def supplement_cpc_adapter_with_render_manifest(
    *,
    corpus_root: str | Path,
    adapter_root: str | Path,
    render_manifest: str | Path,
    output_root: str | Path,
    export_manifest: str | Path | None = None,
    verify_hashes: bool = True,
) -> CpcAdapterSupplementSummary:
    """Write a new adapter bundle with render-backed rows appended locally."""

    corpus = Path(corpus_root).expanduser().resolve()
    adapter = Path(adapter_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    render_path = Path(render_manifest).expanduser().resolve()
    if not corpus.is_dir():
        raise CpcAdapterError(f"corpus root does not exist: {corpus}")
    if not adapter.is_dir():
        raise CpcAdapterError(f"adapter root does not exist: {adapter}")
    if not render_path.is_file():
        raise CpcAdapterError(f"render manifest does not exist: {render_path}")
    if output == corpus or corpus in output.parents:
        raise CpcAdapterError("supplemented adapter output must stay outside the read-only corpus root")
    if output.exists():
        raise CpcAdapterError(f"output already exists; choose a new versioned path: {output}")

    existing_rows = read_corpus_manifest(adapter / "piano_clamp_manifest.csv")
    existing_timing_rows = _read_csv_rows(adapter / "alignment_events.csv")
    existing_recording_ids = {_text(row.get("recording_id", "")) for row in existing_rows}

    export, _, manifest_source = _load_source_manifest(
        corpus=corpus,
        export_manifest=export_manifest,
        requested={_composer_key(row.get("composer", "")) for row in existing_rows},
    )
    render_rows = _read_csv_rows(render_path)
    export_rows = export.to_dict("records")

    works = _unique_table(corpus / "data" / "canonical" / "works.csv", "work_id")
    scores = _unique_table(corpus / "data" / "canonical" / "score_versions.csv", "score_version_id")
    alignments = _unique_table(corpus / "data" / "canonical" / "alignments.csv", "alignment_id")

    export_by_performance_work: dict[tuple[str, str], list[dict[str, str]]] = {}
    requested_alignment_ids: set[str] = set()
    for row in render_rows:
        recording_id = _text(row.get("output_recording_id", ""))
        if recording_id and recording_id not in existing_recording_ids:
            requested_alignment_ids.add(_text(row.get("alignment_id", "")))
    for row in export_rows:
        key = (_text(row.get("performance_id", "")), _text(row.get("work_id", "")))
        export_by_performance_work.setdefault(key, []).append(dict(row))

    alignment_groups: dict[str, dict[str, pd.DataFrame]] = {}
    for alignment_path in ("data/canonical/alignment_events.csv", "data/canonical/alignment_notes.csv"):
        ids = {
            alignment_id
            for alignment_id in requested_alignment_ids
            if (
                alignment_id in alignments.index
                and (
                    _text(alignments.loc[alignment_id].get("granularity")).casefold() == "note"
                ) == alignment_path.endswith("alignment_notes.csv")
            )
        }
        if not ids:
            continue
        path = _resolve_asset(corpus, alignment_path)
        alignment_groups[alignment_path] = _selected_alignment_rows(path, ids)

    hash_cache: dict[Path, str] = {}
    added_rows: list[dict[str, object]] = []
    added_timing_rows: list[dict[str, object]] = []
    unmatched_render_rows = 0

    for render_row in render_rows:
        render_recording_id = _text(render_row.get("output_recording_id", ""))
        if not render_recording_id or render_recording_id in existing_recording_ids:
            continue
        performance_id = _text(render_row.get("performance_id", ""))
        work_id = _text(render_row.get("work_id", ""))
        export_row = _pick_render_export_candidate(
            export_by_performance_work.get((performance_id, work_id), []),
            performance_id=performance_id,
            work_id=work_id,
            score_version_id=_text(render_row.get("score_version_id", "")),
            alignment_id=_text(render_row.get("alignment_id", "")),
        )
        if export_row is None:
            unmatched_render_rows += 1
            continue
        try:
            ids = {
                "composition_id": _validate_id("composition_id", export_row.get("composition_id")),
                "work_id": _validate_id("work_id", work_id),
                "score_version_id": _validate_id("score_version_id", render_row.get("score_version_id")),
                "performance_id": _validate_id("performance_id", performance_id),
                "recording_id": _validate_id("recording_id", render_recording_id),
                "alignment_id": _validate_id("alignment_id", render_row.get("alignment_id")),
            }
            composer = _composer(export_row.get("composer"))
            work = works.loc[ids["work_id"]]
            score_ref = scores.loc[ids["score_version_id"]]
            alignment_ref = alignments.loc[ids["alignment_id"]]
            score_relative = _text(render_row.get("score_path"))
            score_suffix = Path(score_relative).suffix.casefold()
            if score_suffix not in SUPPORTED_SCORE_EXTENSIONS:
                raise CpcAdapterError("render manifest score_path is a metadata locator rather than MusicXML/MXL/MIDI")
            audio_relative = _text(render_row.get("output_audio_path"))
            alignment_store_relative = (
                "data/canonical/alignment_notes.csv"
                if _text(alignment_ref.get("granularity")).casefold() == "note"
                else "data/canonical/alignment_events.csv"
            )
            score_path = _resolve_asset(corpus, score_relative)
            audio_path = _resolve_asset(corpus, audio_relative)
            alignment_store_path = _resolve_asset(corpus, alignment_store_relative)
            for kind, path in (("score", score_path), ("audio", audio_path), ("alignment", alignment_store_path)):
                if not path.is_file():
                    raise CpcAdapterError(f"{kind} asset does not exist: {path}")
            _validate_hash(
                score_ref.get("sha256"),
                field="score_sha256",
                path=score_path,
                cache=hash_cache,
                verify=verify_hashes,
            )
            audio_digest = _text(render_row.get("output_sha256")).casefold()
            if len(audio_digest) != 64:
                raise CpcAdapterError("render manifest output_sha256 is missing or invalid")
            if verify_hashes and _sha256(audio_path, hash_cache) != audio_digest:
                raise CpcAdapterError("render manifest output_sha256 does not match the audio file")
            anchors = alignment_groups.get(alignment_store_relative, {}).get(ids["alignment_id"])
            if anchors is None or anchors.empty:
                raise CpcAdapterError("global alignment store has no rows for supplemented alignment_id")
            granularity = _text(alignment_ref.get("granularity")).casefold()
            duration = _wav_duration(audio_path)
            timings = (
                _note_timings(anchors, score_path=score_path, audio_duration=duration)
                if granularity == "note"
                else _event_timings(anchors, audio_duration=duration)
            )
            licenses = _text(export_row.get("licenses"))
            alignment_quality = _text(render_row.get("alignment_quality")) or _text(export_row.get("alignment_quality"))
            fully_aligned = "false" if "partial" in alignment_quality.casefold() else "true"
            source_dataset = _text(render_row.get("source_dataset")) or _text(export_row.get("original_source_dataset"))
            performance_midi_relative = (
                _text(render_row.get("source_midi_path"))
                or _text(export_row.get("performance_midi_path"))
            )
            added_rows.append(
                {
                    "passage_id": _text(export_row.get("export_row_id")),
                    "composer": composer,
                    "period": _period(composer),
                    "work_id": ids["work_id"],
                    "movement_id": ids["work_id"],
                    "recording_id": ids["recording_id"],
                    "score_path": score_relative,
                    "mxl_path": score_relative if score_suffix in {".mxl", ".musicxml", ".xml"} else "",
                    "audio_path": audio_relative,
                    "midi_path": score_relative if score_suffix in {".mid", ".midi"} else "",
                    "performance_midi_path": performance_midi_relative,
                    "alignment_path": alignment_store_relative,
                    "annotation_path": "",
                    "bar_start": 1,
                    "bar_end": len(timings),
                    "score_format": score_suffix.lstrip("."),
                    "alignment_format": f"cpc_canonical_{granularity}",
                    "rights_status": _rights_status(licenses),
                    "segment_type": "complete_movement",
                    "annotation_type": "",
                    "fully_aligned": fully_aligned,
                    "eligible_for_clamp": "true",
                    "availability_status": "ready",
                    "composition_id": ids["composition_id"],
                    "work_title": _text(render_row.get("work_title")) or _text(work.get("title")) or ids["work_id"],
                    "score_version_id": ids["score_version_id"],
                    "performance_id": ids["performance_id"],
                    "alignment_id": ids["alignment_id"],
                    "audio_origin": "synthetic_rendering",
                    "alignment_granularity": granularity,
                    "alignment_method": _text(alignment_ref.get("method")),
                    "alignment_quality": alignment_quality,
                    "alignment_confidence": "",
                    "source_record_ids": _text(export_row.get("source_record_ids")),
                    "source_dataset": source_dataset,
                    "licenses": licenses,
                }
            )
            for timing in timings:
                added_timing_rows.append(
                    {
                        "recording_id": ids["recording_id"],
                        "movement_id": ids["work_id"],
                        "alignment_id": ids["alignment_id"],
                        **timing,
                        "granularity": granularity,
                        "method": _text(alignment_ref.get("method")),
                        "quality_label": alignment_quality,
                    }
                )
        except (CpcAdapterError, KeyError, OSError, TypeError, ValueError):
            unmatched_render_rows += 1

    combined_rows = existing_rows + added_rows
    combined_timing_rows = existing_timing_rows + added_timing_rows
    combined_rows.sort(key=lambda item: str(item["passage_id"]))
    combined_timing_rows.sort(key=lambda item: (str(item["recording_id"]), int(item["score_measure"])))

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".piano-clamp-cpc-supplement-", dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        shutil.copytree(adapter, stage)
        manifest_output = stage / "piano_clamp_manifest.csv"
        timing_output = stage / "alignment_events.csv"
        cohort_output = stage / "cohort_summary.csv"
        readiness_output = stage / "study_readiness.json"
        metadata_output = stage / "adapter_metadata.json"

        _write_csv(manifest_output, CORPUS_MANIFEST_FIELDS + ADAPTER_EXTRA_FIELDS, combined_rows)
        _write_csv(timing_output, ALIGNMENT_EVENT_FIELDS, combined_timing_rows)
        cohort = (
            pd.DataFrame(combined_rows)
            .groupby(["composer", "audio_origin", "alignment_granularity"])
            .agg(
                recording_count=("recording_id", "nunique"),
                composition_count=("composition_id", "nunique"),
                work_count=("work_id", "nunique"),
            )
            .reset_index()
        )
        cohort.to_csv(cohort_output, index=False)
        counts = (
            pd.DataFrame(combined_rows)
            .groupby(["audio_origin", "composer"])["recording_id"]
            .nunique()
            .to_dict()
        )
        readiness = build_study_readiness_report(
            combined_rows,
            corpus_root=corpus,
            alignment_root=stage,
            authorized_rights=DEFAULT_AUTHORIZED_RIGHTS,
            source_manifest_path=manifest_output,
        )
        readiness["cohort_recording_counts"] = {
            f"{origin}|{composer}": count for (origin, composer), count in counts.items()
        }
        readiness["note"] = "Never mix source recordings and synthetic renderings in a composer contrast."
        readiness_output.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        metadata = {}
        if metadata_output.is_file():
            metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
        metadata.update(
            {
                "adapter_schema": "piano-clamp-cpc-adapter-1.0.0",
                "source_manifest": manifest_source,
                "supplemented_from_render_manifest": str(render_path),
                "supplemented_from_render_manifest_sha256": _sha256(render_path, hash_cache),
                "source_adapter_root": str(adapter),
                "supplement_added_rows": len(added_rows),
                "supplement_unmatched_render_rows": unmatched_render_rows,
                "imported_rows": len(combined_rows),
                "timing_rows": len(combined_timing_rows),
            }
        )
        metadata_output.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if len(read_corpus_manifest(manifest_output)) != len(combined_rows):
            raise CpcAdapterError("supplemented adapter manifest failed its row-count invariant")
        stage.replace(output)

    return CpcAdapterSupplementSummary(
        source_adapter_root=str(adapter),
        output_root=str(output),
        render_manifest=str(render_path),
        added_rows=len(added_rows),
        total_rows=len(combined_rows),
        added_timing_rows=len(added_timing_rows),
        total_timing_rows=len(combined_timing_rows),
        unmatched_render_rows=unmatched_render_rows,
    )
