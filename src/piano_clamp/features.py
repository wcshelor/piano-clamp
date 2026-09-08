"""Transparent, deterministic features derived directly from MusicXML/MXL."""

from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

from .data import read_manifest, resolve_source_files, write_manifest_snapshot
from .provenance import build_data_payload, ensure_input_lock


MUSICXML_EXTENSIONS = {".xml", ".musicxml", ".mxl"}
FEATURE_FIELDS = (
    "measure_count",
    "note_event_count",
    "pitched_note_count",
    "rest_fraction",
    "grace_note_fraction",
    "note_density_per_measure",
    "mean_midi_pitch",
    "midi_pitch_std",
    "midi_pitch_range",
    "pitch_class_entropy_bits",
    "explicit_accidental_fraction",
    "out_of_key_note_fraction",
    "chord_tone_fraction",
    "voice_count",
    "staff_count",
    "sounding_note_quarters",
    "rhythmic_entropy_bits",
    "mean_abs_melodic_interval",
    "large_leap_fraction",
    "ornament_events_per_100_notes",
    "key_fifths_xml",
    "key_mode_xml",
    "meter_beats_xml",
    "meter_beat_type_xml",
)


class FeatureError(RuntimeError):
    """Raised when a passage cannot yield the declared MusicXML feature schema."""


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(node: ET.Element, name: str) -> Iterable[ET.Element]:
    return (child for child in node if _local_name(child.tag) == name)


def _child(node: ET.Element, name: str) -> ET.Element | None:
    return next(_children(node, name), None)


def _text(node: ET.Element | None, name: str, default: str = "") -> str:
    child = _child(node, name) if node is not None else None
    return (child.text or "").strip() if child is not None else default


def _number(node: ET.Element | None, name: str, default: float = 0.0) -> float:
    value = _text(node, name)
    try:
        return float(value)
    except ValueError:
        return default


def _read_musicxml_bytes(path: Path) -> bytes:
    if path.suffix.lower() != ".mxl":
        return path.read_bytes()
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
            score_name: str | None = None
            if "META-INF/container.xml" in names:
                container = ET.fromstring(archive.read("META-INF/container.xml"))
                for element in container.iter():
                    if _local_name(element.tag) == "rootfile":
                        candidate = element.attrib.get("full-path")
                        if candidate in names:
                            score_name = candidate
                            break
            if score_name is None:
                score_name = next(
                    (
                        name
                        for name in archive.namelist()
                        if name.lower().endswith((".musicxml", ".xml"))
                        and not name.startswith("META-INF/")
                    ),
                    None,
                )
            if score_name is None:
                raise FeatureError(f"MXL archive contains no MusicXML score: {path}")
            return archive.read(score_name)
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise FeatureError(f"invalid MXL archive: {path}") from exc


def _entropy(values: Iterable[object]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _key_pitch_classes(fifths: int, mode: str) -> set[int]:
    normalized_mode = mode.lower()
    tonic = ((9 if normalized_mode == "minor" else 0) + 7 * fifths) % 12
    intervals = (0, 2, 3, 5, 7, 8, 10) if normalized_mode == "minor" else (0, 2, 4, 5, 7, 9, 11)
    return {(tonic + interval) % 12 for interval in intervals}


def extract_musicxml_features(path: str | Path) -> dict[str, int | float | str]:
    """Extract a documented feature vector from one pre-segmented score."""

    source = Path(path)
    if source.suffix.lower() not in MUSICXML_EXTENSIONS:
        raise FeatureError(f"MusicXML feature extraction does not support {source.suffix}: {source}")
    try:
        root = ET.fromstring(_read_musicxml_bytes(source))
    except ET.ParseError as exc:
        raise FeatureError(f"invalid MusicXML: {source}: {exc}") from exc
    if _local_name(root.tag) != "score-partwise":
        raise FeatureError(f"only score-partwise MusicXML is supported: {source}")

    step_pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    pitches: list[int] = []
    durations: list[float] = []
    melodic_by_voice: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    part_measure_counts: list[int] = []
    note_events = rests = grace_notes = chord_tones = explicit_accidentals = 0
    ornament_events = 0
    voices: set[tuple[str, str]] = set()
    staves: set[tuple[str, str]] = set()
    key_fifths: int | None = None
    key_mode = ""
    meter_beats = ""
    meter_beat_type = ""

    for part_index, part in enumerate(_children(root, "part"), start=1):
        part_id = part.attrib.get("id", f"part-{part_index}")
        measures = list(_children(part, "measure"))
        part_measure_counts.append(len(measures))
        divisions = 1.0
        for measure in measures:
            attributes = _child(measure, "attributes")
            if attributes is not None:
                parsed_divisions = _number(attributes, "divisions", divisions)
                divisions = parsed_divisions if parsed_divisions > 0 else divisions
                key = _child(attributes, "key")
                if key is not None and key_fifths is None:
                    try:
                        key_fifths = int(_text(key, "fifths", "0"))
                    except ValueError:
                        key_fifths = 0
                    key_mode = _text(key, "mode", "major") or "major"
                time = _child(attributes, "time")
                if time is not None and not meter_beats:
                    meter_beats = _text(time, "beats")
                    meter_beat_type = _text(time, "beat-type")

            for note in _children(measure, "note"):
                note_events += 1
                is_rest = _child(note, "rest") is not None
                is_grace = _child(note, "grace") is not None
                is_chord = _child(note, "chord") is not None
                if is_rest:
                    rests += 1
                if is_grace:
                    grace_notes += 1

                voice = _text(note, "voice", "1") or "1"
                staff = _text(note, "staff", "1") or "1"
                voices.add((part_id, voice))
                staves.add((part_id, staff))

                pitch = _child(note, "pitch")
                if pitch is None:
                    continue
                step = _text(pitch, "step").upper()
                try:
                    octave = int(_text(pitch, "octave"))
                    alter = int(round(float(_text(pitch, "alter", "0"))))
                    midi = 12 * (octave + 1) + step_pc[step] + alter
                except (KeyError, ValueError):
                    continue
                pitches.append(midi)
                if is_chord:
                    chord_tones += 1
                else:
                    melodic_by_voice[(part_id, voice, staff)].append(midi)
                if _child(note, "accidental") is not None:
                    explicit_accidentals += 1
                if not is_grace and not is_chord:
                    duration = _number(note, "duration") / divisions
                    if duration > 0:
                        durations.append(duration)
                notations = _child(note, "notations")
                if notations is not None:
                    ornament_names = {
                        "trill-mark",
                        "turn",
                        "inverted-turn",
                        "mordent",
                        "inverted-mordent",
                        "shake",
                    }
                    ornament_events += sum(
                        1 for element in notations.iter() if _local_name(element.tag) in ornament_names
                    )

    if not pitches:
        raise FeatureError(f"MusicXML contains no parseable pitched notes: {source}")
    measure_count = max(part_measure_counts, default=0)
    intervals = [
        abs(current - previous)
        for sequence in melodic_by_voice.values()
        for previous, current in zip(sequence, sequence[1:])
    ]
    pitch_mean = mean(pitches)
    pitch_std = math.sqrt(mean([(pitch - pitch_mean) ** 2 for pitch in pitches]))
    in_key = (
        _key_pitch_classes(key_fifths, key_mode)
        if key_fifths is not None and key_mode.lower() in {"major", "minor"}
        else None
    )
    out_of_key = (
        sum(1 for pitch in pitches if pitch % 12 not in in_key) / len(pitches)
        if in_key is not None
        else float("nan")
    )
    return {
        "measure_count": measure_count,
        "note_event_count": note_events,
        "pitched_note_count": len(pitches),
        "rest_fraction": rests / note_events if note_events else 0.0,
        "grace_note_fraction": grace_notes / note_events if note_events else 0.0,
        "note_density_per_measure": len(pitches) / measure_count if measure_count else 0.0,
        "mean_midi_pitch": pitch_mean,
        "midi_pitch_std": pitch_std,
        "midi_pitch_range": max(pitches) - min(pitches),
        "pitch_class_entropy_bits": _entropy(pitch % 12 for pitch in pitches),
        "explicit_accidental_fraction": explicit_accidentals / len(pitches),
        "out_of_key_note_fraction": out_of_key,
        "chord_tone_fraction": chord_tones / len(pitches),
        "voice_count": len(voices),
        "staff_count": len(staves),
        "sounding_note_quarters": sum(durations),
        "rhythmic_entropy_bits": _entropy(round(duration, 6) for duration in durations),
        "mean_abs_melodic_interval": mean(intervals) if intervals else 0.0,
        "large_leap_fraction": sum(interval >= 5 for interval in intervals) / len(intervals) if intervals else 0.0,
        "ornament_events_per_100_notes": 100 * ornament_events / len(pitches),
        "key_fifths_xml": key_fifths if key_fifths is not None else "",
        "key_mode_xml": key_mode,
        "meter_beats_xml": meter_beats,
        "meter_beat_type_xml": meter_beat_type,
    }


def extract_features(config: Mapping[str, Any]) -> Path:
    """Extract one MusicXML feature row per manifest passage."""

    rows = read_manifest(config["paths"]["manifest"])
    sources = resolve_source_files(rows, config["paths"]["data_root"])
    unsupported = [
        row["passage_id"]
        for row in rows
        if sources[row["passage_id"]].suffix.lower() not in MUSICXML_EXTENSIONS
    ]
    if unsupported:
        raise FeatureError(
            "MusicXML features require .xml, .musicxml, or .mxl passages; unsupported IDs: "
            + ", ".join(unsupported[:10])
        )

    from .embeddings import initialize_run

    run_dir = initialize_run(config)
    write_manifest_snapshot(rows, run_dir / "manifest.snapshot.csv")
    data_payload = build_data_payload(rows, sources)
    ensure_input_lock(run_dir, "score_data", data_payload)

    records = []
    for row in rows:
        record: dict[str, Any] = {
            "passage_id": row["passage_id"],
            "composer": row["composer"],
            "work_title": row["work_title"],
            "movement": row["movement"],
            "start_bar": row["start_bar"],
            "end_bar": row["end_bar"],
            "manifest_key": row["key"],
            "manifest_meter": row["meter"],
        }
        for field in ("composition_id", "work_id", "score_version_id", "source_dataset"):
            if field in row:
                record[field] = row[field]
        record.update(extract_musicxml_features(sources[row["passage_id"]]))
        records.append(record)

    output = run_dir / "tables" / "musicxml_features.csv"
    fields = tuple(records[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return output
