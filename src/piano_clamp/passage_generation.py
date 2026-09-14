"""Deterministic score/performance passage records and local segment extraction."""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .data_loading import AUDIO_EXTENSIONS, SYMBOLIC_EXTENSIONS, read_lookup, resolve_corpus_asset
from .prepare import crop_musicxml, mxl_musicxml


class PassageGenerationError(RuntimeError):
    """Raised when a requested passage cannot be grounded in source metadata."""


PASSAGE_FIELDS = (
    "passage_id",
    "composer",
    "period",
    "composition_id",
    "work_id",
    "movement_id",
    "recording_id",
    "performer",
    "bar_start",
    "bar_end",
    "score_path",
    "audio_path",
    "midi_path",
    "performance_midi_path",
    "alignment_path",
    "annotation_path",
    "rights_status",
    "audio_origin",
    "passage_generation_method",
    "embedding_modality",
    "source_material",
    "source_path",
    "onset_seconds",
    "offset_seconds",
    "content_hash",
    "embedding_row",
    "embedding_dimension",
    "status",
    "error_message",
)


def _slug(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text or "none"


def stable_passage_id(record: Mapping[str, Any], start: int, end: int) -> str:
    composer = _slug(str(record.get("composer", "")))
    recording = _slug(str(record.get("recording_id") or "score"))
    collection = _slug(str(record.get("source_collection") or "corpus"))
    return "_".join(
        (
            composer,
            _slug(str(record.get("work_id", "work"))),
            _slug(str(record.get("movement_id", "movement"))),
            collection,
            recording,
            f"{start:04d}",
            f"{end:04d}",
        )
    )


def parse_bar_ranges(values: Sequence[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for value in values:
        match = re.fullmatch(r"\s*(\d+)\s*[-:]\s*(\d+)\s*", value)
        if not match:
            raise PassageGenerationError(f"invalid bar range {value!r}; expected START-END")
        start, end = int(match.group(1)), int(match.group(2))
        if start < 1 or end < start:
            raise PassageGenerationError(f"invalid bar range {value!r}")
        ranges.append((start, end))
    return ranges


def _annotation_ranges(path: Path, *, kind: str, lower: int, upper: int) -> list[tuple[int, int]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    boundaries: list[int] = []
    for row in rows:
        measure_text = str(row.get("mn") or row.get("mc") or "").split(".")[0]
        try:
            measure = int(measure_text)
        except ValueError:
            continue
        if kind == "phrase":
            marker = str(row.get("phraseend", ""))
            if "}" not in marker:
                continue
        else:
            marker = str(row.get("cadence", "")).strip()
            if not marker:
                continue
        if lower <= measure <= upper:
            boundaries.append(measure)
    boundaries = sorted(set(boundaries))
    start = lower
    ranges: list[tuple[int, int]] = []
    for boundary in boundaries:
        if boundary >= start:
            ranges.append((start, boundary))
            start = boundary + 1
    if start <= upper and kind == "phrase":
        ranges.append((start, upper))
    return ranges


def generate_passage_records(
    rows: Iterable[Mapping[str, str]],
    *,
    corpus_root: str | Path,
    method: str,
    window_bars: int | None = None,
    stride_bars: int | None = None,
    user_ranges: Sequence[tuple[int, int]] = (),
) -> list[dict[str, Any]]:
    """Generate provenance-only passage records without touching external data."""

    if method not in {"complete_movement", "fixed_bar_window", "phrase", "cadence", "user_range"}:
        raise PassageGenerationError(f"unsupported passage method: {method}")
    if method == "fixed_bar_window" and window_bars not in {4, 8, 16}:
        raise PassageGenerationError("fixed windows must contain 4, 8, or 16 bars")
    if method == "fixed_bar_window" and stride_bars is not None and stride_bars < 1:
        raise PassageGenerationError("fixed-window stride must be positive")
    root = Path(corpus_root)
    recording_lookup = read_lookup(root / "manifests" / "recordings.csv", "recording_id")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in sorted(
        rows,
        key=lambda row: (
            row.get("composer", ""),
            row.get("work_id", ""),
            row.get("movement_id", ""),
            row.get("recording_id", ""),
            row.get("passage_id", ""),
        ),
    ):
        lower, upper = int(source["bar_start"]), int(source["bar_end"])
        if method == "complete_movement":
            ranges = [(lower, upper)]
        elif method == "fixed_bar_window":
            assert window_bars is not None
            step = stride_bars or window_bars
            ranges = [
                (start, min(start + window_bars - 1, upper))
                for start in range(lower, upper + 1, step)
                if min(start + window_bars - 1, upper) - start + 1 == window_bars
            ]
        elif method == "user_range":
            ranges = [(start, end) for start, end in user_ranges if lower <= start <= end <= upper]
        else:
            annotation_value = source.get("annotation_path", "")
            annotation = resolve_corpus_asset(root, annotation_value) if annotation_value else Path()
            ranges = _annotation_ranges(annotation, kind=method, lower=lower, upper=upper)

        detail = recording_lookup.get(source.get("recording_id", ""), {})
        source_collection = detail.get("source_collection", "")
        for start, end in ranges:
            context = {**source, "source_collection": source_collection}
            passage_id = stable_passage_id(context, start, end)
            if passage_id in seen:
                continue
            seen.add(passage_id)
            score_path = source.get("mxl_path") or source.get("score_path", "")
            output.append(
                {
                    "passage_id": passage_id,
                    "composer": source.get("composer", ""),
                    "period": source.get("period", ""),
                    "composition_id": source.get("composition_id", ""),
                    "work_id": source.get("work_id", ""),
                    "movement_id": source.get("movement_id", ""),
                    "recording_id": source.get("recording_id", ""),
                    "performer": detail.get("performer", ""),
                    "bar_start": start,
                    "bar_end": end,
                    "score_path": score_path,
                    "audio_path": source.get("audio_path", ""),
                    "midi_path": source.get("midi_path", ""),
                    "performance_midi_path": source.get("performance_midi_path", ""),
                    "alignment_path": source.get("alignment_path", ""),
                    "annotation_path": source.get("annotation_path", ""),
                    "rights_status": source.get("rights_status", ""),
                    "audio_origin": source.get("audio_origin", ""),
                    "passage_generation_method": (
                        (
                            f"fixed_{window_bars}_bars_stride_{step}"
                            if method == "fixed_bar_window" and step != window_bars
                            else f"fixed_{window_bars}_bars"
                        )
                        if method == "fixed_bar_window"
                        else method
                    ),
                    "embedding_modality": "",
                    "source_material": "",
                    "source_path": "",
                    "onset_seconds": "",
                    "offset_seconds": "",
                    "status": "pending",
                    "error_message": "",
                }
            )
    return output


def load_alignment_times(alignment_root: str | Path) -> dict[tuple[str, str], list[dict[str, str]]]:
    path = Path(alignment_root)
    if path.is_dir():
        path = path / "alignment_events.csv"
    if not path.is_file():
        return {}
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            grouped[(str(row.get("recording_id", "")), str(row.get("movement_id", "")))].append(dict(row))
    return grouped


def aligned_measure_coverage(
    record: Mapping[str, Any],
    events: Mapping[tuple[str, str], Sequence[Mapping[str, str]]],
) -> dict[str, Any]:
    """Return exact measure coverage for one requested audio passage."""

    start = int(record["bar_start"])
    end = int(record["bar_end"])
    expected = list(range(start, end + 1))
    candidates = events.get((str(record.get("recording_id", "")), str(record.get("movement_id", ""))), ())
    by_measure: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for event in candidates:
        try:
            measure = int(float(str(event.get("score_measure", ""))))
            onset = float(str(event.get("onset_seconds", "")))
            offset = float(str(event.get("offset_seconds", "")))
        except ValueError:
            continue
        if start <= measure <= end and offset > onset:
            by_measure[measure].append((onset, offset))
    present = sorted(by_measure)
    missing = [measure for measure in expected if measure not in by_measure]
    onset = min((min(item[0] for item in values) for values in by_measure.values()), default=None)
    offset = max((max(item[1] for item in values) for values in by_measure.values()), default=None)
    return {
        "expected_measures": expected,
        "present_measures": present,
        "missing_measures": missing,
        "covered_measure_count": len(present),
        "expected_measure_count": len(expected),
        "fully_covered": not missing and onset is not None and offset is not None and offset > onset,
        "onset_seconds": onset,
        "offset_seconds": offset,
    }


def aligned_time_range(
    record: Mapping[str, Any],
    events: Mapping[tuple[str, str], Sequence[Mapping[str, str]]],
) -> tuple[float, float] | None:
    coverage = aligned_measure_coverage(record, events)
    if not coverage["fully_covered"]:
        return None
    onset = coverage["onset_seconds"]
    offset = coverage["offset_seconds"]
    assert onset is not None and offset is not None
    return (onset, offset) if offset > onset else None


def materialize_symbolic_passage(
    record: Mapping[str, Any],
    corpus_root: str | Path,
    destination: Path,
    *,
    source_material: str = "score",
    events: Mapping[tuple[str, str], Sequence[Mapping[str, str]]] | None = None,
) -> Path:
    """Create a temporary supported symbolic segment, leaving the corpus untouched."""

    if source_material not in {"score", "performance_midi"}:
        raise PassageGenerationError(f"unsupported symbolic source material: {source_material}")
    if source_material == "performance_midi":
        source_value = str(record.get("performance_midi_path", ""))
    else:
        source_value = str(record.get("score_path", ""))
        midi_value = str(record.get("midi_path", ""))
        if Path(source_value).suffix.lower() not in SYMBOLIC_EXTENSIONS and midi_value:
            source_value = midi_value
    if not source_value:
        raise PassageGenerationError(f"no {source_material} symbolic source is recorded")
    source = resolve_corpus_asset(corpus_root, source_value)
    if not source.is_file():
        raise PassageGenerationError(f"{source_material} symbolic source is not available locally")
    suffix = source.suffix.lower()
    method = str(record.get("passage_generation_method", ""))
    destination.mkdir(parents=True, exist_ok=True)
    if source_material == "performance_midi":
        if suffix not in {".mid", ".midi"}:
            raise PassageGenerationError(f"unsupported performance MIDI source: {suffix}")
        if method == "complete_movement":
            target = destination / f"{record['passage_id']}{suffix}"
            target.symlink_to(source.resolve())
            return target
        if events is None:
            raise PassageGenerationError("performance MIDI passage cropping requires alignment events")
        times = aligned_time_range(record, events)
        if times is None:
            raise PassageGenerationError("no aligned onset/offset events cover this bar range")
        target = destination / f"{record['passage_id']}.mid"
        crop_midi_time_range(source, target, onset_seconds=times[0], offset_seconds=times[1])
        return target
    if suffix in {".mid", ".midi"}:
        if method == "complete_movement":
            target = destination / f"{record['passage_id']}{suffix}"
            target.symlink_to(source.resolve())
            return target
        try:
            from music21 import converter

            score = converter.parse(str(source))
            cropped = score.measures(int(record["bar_start"]), int(record["bar_end"]))
            if not list(getattr(cropped, "parts", [])) and not list(cropped.recurse().notes):
                raise PassageGenerationError("MIDI crop contains no score parts or notes")
            target = destination / f"{record['passage_id']}.musicxml"
            cropped.write("musicxml", fp=str(target))
            if not target.is_file():
                raise PassageGenerationError("music21 did not write the cropped MIDI passage")
            return target
        except ImportError as exc:
            raise PassageGenerationError(
                "music21 is required for bar-accurate MIDI passage cropping"
            ) from exc
        except PassageGenerationError:
            raise
        except Exception as exc:
            raise PassageGenerationError(f"MIDI passage crop failed: {exc}") from exc
    if suffix not in {".mxl", ".musicxml", ".xml"}:
        raise PassageGenerationError(f"unsupported symbolic passage source: {suffix}")
    raw = mxl_musicxml(source.read_bytes()) if suffix == ".mxl" else source.read_bytes()
    start = int(record["bar_start"])
    length = int(record["bar_end"]) - start + 1
    try:
        cropped, _, _ = crop_musicxml(raw, start_index=start - 1, window_bars=length)
    except Exception as exc:
        raise PassageGenerationError(f"MusicXML passage crop failed: {exc}") from exc
    target = destination / f"{record['passage_id']}.musicxml"
    target.write_bytes(cropped)
    return target


def crop_midi_time_range(
    source: str | Path,
    target: str | Path,
    *,
    onset_seconds: float,
    offset_seconds: float,
) -> None:
    """Crop a MIDI file by performance time while preserving MIDI event payloads."""

    if offset_seconds <= onset_seconds:
        raise PassageGenerationError("MIDI crop time range is empty")
    try:
        import mido
    except ImportError as exc:
        raise PassageGenerationError("mido is required for performance MIDI passage cropping") from exc

    mid = mido.MidiFile(str(source))
    cropped = mido.MidiFile(
        type=mid.type,
        ticks_per_beat=mid.ticks_per_beat,
        charset=getattr(mid, "charset", "latin1"),
    )
    tick_rate = mid.ticks_per_beat
    kept_note_events = 0
    for track in mid.tracks:
        output_track = mido.MidiTrack()
        cropped.tracks.append(output_track)
        absolute_seconds = 0.0
        tempo = 500000
        last_seconds = onset_seconds
        for message in track:
            absolute_seconds += mido.tick2second(int(message.time), tick_rate, tempo)
            seconds = absolute_seconds
            if message.is_meta and message.type not in {"set_tempo", "time_signature", "key_signature", "end_of_track"}:
                continue
            if seconds < onset_seconds:
                if message.type in {"set_tempo", "time_signature", "key_signature"}:
                    copied = message.copy(time=0)
                    output_track.append(copied)
                if message.type == "set_tempo":
                    tempo = int(message.tempo)
                continue
            if seconds > offset_seconds:
                break
            delta_seconds = max(0.0, seconds - last_seconds)
            copied = message.copy(time=max(0, round(mido.second2tick(delta_seconds, tick_rate, tempo))))
            output_track.append(copied)
            last_seconds = seconds
            if message.type in {"note_on", "note_off"}:
                kept_note_events += 1
            if message.type == "set_tempo":
                tempo = int(message.tempo)
        if not output_track or output_track[-1].type != "end_of_track":
            output_track.append(mido.MetaMessage("end_of_track", time=0))
    if kept_note_events == 0:
        raise PassageGenerationError("performance MIDI crop contains no note events")
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    cropped.save(str(target))


def rights_authorized(status: str, allowed_values: Sequence[str]) -> bool:
    normalized = status.strip().casefold()
    return any(value.strip().casefold() == normalized for value in allowed_values)


def load_audio(path: str | Path, *, sample_rate: int, mono: bool = True) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise PassageGenerationError("soundfile is required for audio chunking") from exc
    try:
        waveform, source_rate = sf.read(str(path), always_2d=True, dtype="float32")
    except Exception as exc:
        raise PassageGenerationError(f"cannot decode local audio: {exc}") from exc
    if waveform.size == 0:
        raise PassageGenerationError("audio file is empty")
    if mono:
        waveform = waveform.mean(axis=1)
    elif waveform.shape[1] == 1:
        waveform = waveform[:, 0]
    if source_rate != sample_rate:
        old = np.linspace(0.0, 1.0, num=waveform.shape[0], endpoint=False)
        new_size = max(1, round(waveform.shape[0] * sample_rate / source_rate))
        new = np.linspace(0.0, 1.0, num=new_size, endpoint=False)
        if waveform.ndim == 1:
            waveform = np.interp(new, old, waveform).astype(np.float32)
        else:
            waveform = np.stack(
                [np.interp(new, old, waveform[:, channel]) for channel in range(waveform.shape[1])],
                axis=1,
            ).astype(np.float32)
    return np.asarray(waveform, dtype=np.float32), sample_rate


def write_wav(path: str | Path, waveform: np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise PassageGenerationError("soundfile is required for audio output") from exc
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), waveform, sample_rate, subtype="PCM_16")


def chunk_audio(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    chunk_seconds: float,
    overlap_seconds: float = 0.0,
) -> list[tuple[float, float, np.ndarray]]:
    if chunk_seconds <= 0 or overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise PassageGenerationError("audio chunk and overlap durations are invalid")
    chunk_samples = round(chunk_seconds * sample_rate)
    step = round((chunk_seconds - overlap_seconds) * sample_rate)
    chunks: list[tuple[float, float, np.ndarray]] = []
    for start in range(0, waveform.shape[0], step):
        end = min(start + chunk_samples, waveform.shape[0])
        if end <= start:
            break
        chunks.append((start / sample_rate, end / sample_rate, waveform[start:end]))
        if end == waveform.shape[0]:
            break
    return chunks


def extract_aligned_audio_passage(
    record: Mapping[str, Any],
    *,
    corpus_root: str | Path,
    events: Mapping[tuple[str, str], Sequence[Mapping[str, str]]],
    destination: Path,
    sample_rate: int,
) -> tuple[Path, float, float]:
    audio_value = str(record.get("audio_path", ""))
    if not audio_value or Path(audio_value).suffix.lower() not in AUDIO_EXTENSIONS:
        raise PassageGenerationError("no supported local audio source is recorded")
    times = aligned_time_range(record, events)
    if times is None:
        raise PassageGenerationError("no aligned onset/offset events cover this bar range")
    source = resolve_corpus_asset(corpus_root, audio_value)
    if not source.is_file():
        raise PassageGenerationError("aligned audio source is not available locally")
    waveform, rate = load_audio(source, sample_rate=sample_rate, mono=True)
    onset, offset = times
    start_sample = max(0, round(onset * rate))
    end_sample = min(waveform.shape[0], round(offset * rate))
    if end_sample <= start_sample:
        raise PassageGenerationError("aligned audio range is empty after clipping")
    target = destination / f"{record['passage_id']}.wav"
    write_wav(target, waveform[start_sample:end_sample], rate)
    return target, onset, offset
