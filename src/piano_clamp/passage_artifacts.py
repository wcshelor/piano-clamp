"""Artifacts for inspecting cached passage windows and failures."""

from __future__ import annotations

import csv
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .musescore_conversion import MuseScoreConversionError, probe_musescore_version


FAILED_FIELDS = (
    "passage_id",
    "composer",
    "composition_id",
    "work_id",
    "movement_id",
    "recording_id",
    "bar_start",
    "bar_end",
    "passage_generation_method",
    "embedding_modality",
    "source_material",
    "source_path",
    "status",
    "error_message",
    "content_hash",
    "prepared_path",
)

REVIEW_FIELDS = (
    "passage_id",
    "composer",
    "composition_id",
    "work_id",
    "movement_id",
    "recording_id",
    "bar_start",
    "bar_end",
    "passage_generation_method",
    "embedding_modality",
    "source_material",
    "source_path",
    "status",
    "error_message",
    "content_hash",
    "prepared_source_path",
    "review_copy_path",
    "png_preview_paths",
)


def _render_pianoroll_preview(source: Path, output: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from music21 import chord, converter, note
    except ImportError as exc:
        raise MuseScoreConversionError(
            "piano-roll preview rendering requires matplotlib and music21"
        ) from exc

    score = converter.parse(str(source))
    pitched_events: list[tuple[float, float, int]] = []
    measure_offsets: list[float] = []
    measure_labels: list[str] = []
    for measure in score.recurse().getElementsByClass("Measure"):
        measure_offsets.append(float(measure.offset))
        measure_labels.append(str(measure.number))
    for event in score.flatten().notesAndRests:
        duration = float(event.quarterLength or 0.0)
        start = float(event.offset or 0.0)
        if isinstance(event, note.Rest):
            continue
        if isinstance(event, note.Note):
            pitched_events.append((start, max(duration, 0.25), int(event.pitch.midi)))
            continue
        if isinstance(event, chord.Chord):
            for pitch in event.pitches:
                pitched_events.append((start, max(duration, 0.25), int(pitch.midi)))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig_width = 8.0
    if pitched_events:
        end_time = max(start + duration for start, duration, _ in pitched_events)
        fig_width = min(20.0, max(8.0, end_time * 0.6))
    fig, axis = plt.subplots(figsize=(fig_width, 4.0), constrained_layout=True)
    if pitched_events:
        for start, duration, pitch in pitched_events:
            axis.hlines(pitch, start, start + duration, linewidth=3.0, color="#1f77b4")
        pitches = [pitch for _, _, pitch in pitched_events]
        axis.set_ylim(min(pitches) - 2, max(pitches) + 2)
    else:
        axis.text(0.5, 0.5, "No pitched note events parsed", ha="center", va="center", transform=axis.transAxes)
    for offset, label in zip(measure_offsets, measure_labels):
        axis.axvline(offset, color="#d9d9d9", linewidth=1.0, linestyle="--")
        axis.text(offset, 1.01, label, fontsize=8, ha="left", va="bottom", transform=axis.get_xaxis_transform())
    axis.set_title(source.name)
    axis.set_xlabel("Quarter-note offset")
    axis.set_ylabel("MIDI pitch")
    axis.grid(axis="y", color="#efefef", linewidth=0.8)
    axis.set_axisbelow(True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return [output]


def _render_musescore_preview(source: Path, output_dir: Path, musescore_binary: str) -> list[Path]:
    stem = source.stem
    expected = output_dir / f"{stem}.png"
    before = {path.name for path in output_dir.glob(f"{stem}*.png")}
    try:
        subprocess.run(
            [musescore_binary, "-o", str(expected), str(source)],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise MuseScoreConversionError(f"MuseScore CLI is not available: {musescore_binary}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stdout or exc.stderr or "").strip()
        raise MuseScoreConversionError(
            f"PNG render failed for {source.name}: {detail or exc}"
        ) from exc
    after = sorted(path for path in output_dir.glob(f"{stem}*.png") if path.name not in before)
    if expected.is_file() and expected not in after:
        after = [expected, *after]
    if not after:
        raise MuseScoreConversionError(f"PNG render produced no output for {source.name}")
    return after


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else str(row.get(field, "")) for field in fields})
    return path


def write_failed_passage_rows(
    destination: str | Path,
    *,
    records: Sequence[Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, str]],
) -> Path:
    rows: list[dict[str, Any]] = []
    by_id = {str(record.get("passage_id", "")): dict(record) for record in records}
    for passage_id, state in sorted(status_map.items()):
        if state.get("status") != "failed":
            continue
        record = dict(by_id.get(passage_id, {}))
        record.update(
            {
                "passage_id": passage_id,
                "status": state.get("status", ""),
                "error_message": state.get("error_message", ""),
                "content_hash": state.get("content_hash", ""),
                "prepared_path": state.get("prepared_path", ""),
            }
        )
        rows.append(record)
    return _write_csv(Path(destination), rows, FAILED_FIELDS)


def export_review_passages(
    destination: str | Path,
    *,
    records: Sequence[Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, str]],
    prepared_root: str | Path,
    status_filter: str = "all",
    passage_ids: Sequence[str] = (),
    limit: int | None = None,
    render_png: bool = True,
    musescore_binary: str = "mscore",
    preview_renderer: str = "auto",
) -> dict[str, Path | int]:
    output = Path(destination)
    prepared = Path(prepared_root)
    wanted_ids = set(str(item) for item in passage_ids)
    review_rows: list[dict[str, Any]] = []
    copied = 0
    png_count = 0
    musescore_version = ""
    if preview_renderer not in {"auto", "musescore", "pianoroll"}:
        raise ValueError(f"unknown preview renderer: {preview_renderer}")
    if render_png and preview_renderer == "musescore":
        musescore_version = probe_musescore_version(musescore_binary)
    if render_png and preview_renderer == "auto":
        try:
            musescore_version = probe_musescore_version(musescore_binary)
        except MuseScoreConversionError:
            musescore_version = ""
    for record in sorted(records, key=lambda row: str(row.get("passage_id", ""))):
        passage_id = str(record.get("passage_id", ""))
        if wanted_ids and passage_id not in wanted_ids:
            continue
        state = dict(status_map.get(passage_id, {}))
        status = str(state.get("status", record.get("status", "")))
        if status_filter != "all" and status != status_filter:
            continue
        prepared_value = str(state.get("prepared_path", "")).strip()
        source_path = (prepared.parent / prepared_value).resolve() if prepared_value else None
        if not source_path or not source_path.is_file():
            continue
        review_dir = output / "prepared"
        review_dir.mkdir(parents=True, exist_ok=True)
        target = review_dir / source_path.name
        if not target.exists():
            shutil.copy2(source_path, target)
            copied += 1
        png_paths: list[str] = []
        if render_png:
            png_dir = output / "png"
            png_dir.mkdir(parents=True, exist_ok=True)
            stem = target.stem
            expected = png_dir / f"{stem}.png"
            if preview_renderer == "pianoroll":
                previews = _render_pianoroll_preview(target, expected)
            elif preview_renderer == "musescore":
                previews = _render_musescore_preview(target, png_dir, musescore_binary)
            else:
                try:
                    previews = _render_musescore_preview(target, png_dir, musescore_binary)
                except MuseScoreConversionError:
                    previews = _render_pianoroll_preview(target, expected)
            png_count += len(previews)
            png_paths = [str(path.relative_to(output)) for path in previews]
        review_rows.append(
            {
                **dict(record),
                "status": status,
                "error_message": state.get("error_message", record.get("error_message", "")),
                "content_hash": state.get("content_hash", record.get("content_hash", "")),
                "prepared_source_path": str(source_path),
                "review_copy_path": str(target.relative_to(output)),
                "png_preview_paths": ";".join(png_paths),
            }
        )
        if limit is not None and len(review_rows) >= limit:
            break
    manifest = _write_csv(output / "review_manifest.csv", review_rows, REVIEW_FIELDS)
    return {
        "manifest_path": manifest,
        "prepared_directory": output / "prepared",
        "png_directory": output / "png",
        "exported_rows": len(review_rows),
        "copied_files": copied,
        "png_files": png_count,
        "musescore_version": musescore_version,
    }
