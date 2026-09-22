#!/usr/bin/env python3
"""Build a score-only Piano CLaMP adapter from CPC canonical score tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from piano_clamp.data_loading import CORPUS_MANIFEST_FIELDS, read_corpus_manifest  # noqa: E402
from piano_clamp.prepare import measure_count, mxl_musicxml  # noqa: E402


SUPPORTED_SCORE_EXTENSIONS = {".mid", ".midi", ".mxl", ".musicxml", ".xml"}
FORMAT_PRIORITY = {
    ".musicxml": 0,
    ".xml": 1,
    ".mxl": 2,
    ".mid": 3,
    ".midi": 3,
}


class SymbolicAdapterError(ValueError):
    """Raised when a symbolic CPC adapter cannot be built."""


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _ascii(value: object) -> str:
    return unicodedata.normalize("NFKD", _text(value)).encode("ascii", "ignore").decode().casefold()


def _composer(value: object) -> str:
    key = _ascii(value)
    if "chopin" in key:
        return "Frédéric Chopin"
    if "mozart" in key:
        return "Wolfgang Amadeus Mozart"
    if "bach" in key:
        return "Johann Sebastian Bach"
    return _text(value)


def _composer_key(value: object) -> str:
    composer = _composer(value)
    if "Chopin" in composer:
        return "Chopin"
    if "Mozart" in composer:
        return "Mozart"
    if "Bach" in composer:
        return "Bach"
    return composer


def _period(composer: str) -> str:
    if "Chopin" in composer:
        return "Romantic"
    if "Mozart" in composer:
        return "Classical"
    if "Bach" in composer:
        return "Baroque"
    return ""


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SymbolicAdapterError(f"CSV file does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _resolve_asset(corpus_root: Path, relative: object) -> Path:
    rel = Path(_text(relative))
    if not rel or rel.is_absolute() or ".." in rel.parts:
        raise SymbolicAdapterError(f"CPC asset path must stay repository-relative: {relative}")
    resolved = (corpus_root / rel).resolve(strict=False)
    if resolved != corpus_root and corpus_root not in resolved.parents:
        raise SymbolicAdapterError(f"CPC asset path escapes the corpus root: {relative}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release(corpus_root: Path) -> str:
    releases = sorted((corpus_root / "manifests" / "releases").glob("*.json"))
    for path in reversed(releases):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        value = _text(payload.get("release")) or _text(payload.get("corpus_release"))
        if value:
            return value
    return releases[-1].stem if releases else "unknown"


def _stable_id(*parts: object) -> str:
    payload = "\x1f".join(_text(part) for part in parts).encode("utf-8")
    return "sym_" + hashlib.sha1(payload).hexdigest()[:12]


def _score_measure_count(path: Path) -> int:
    suffix = path.suffix.casefold()
    if suffix in {".mxl", ".musicxml", ".xml"}:
        payload = mxl_musicxml(path.read_bytes()) if suffix == ".mxl" else path.read_bytes()
        return measure_count(payload)
    if suffix in {".mid", ".midi"}:
        try:
            from music21 import converter, stream
        except ImportError as exc:
            raise SymbolicAdapterError("music21 is required to count MIDI measures") from exc
        try:
            score = converter.parse(str(path))
        except Exception as exc:
            raise SymbolicAdapterError(f"could not parse MIDI score {path}: {exc}") from exc
        parts = list(getattr(score, "parts", [])) or [score]
        measures: list[Any] = []
        for part in parts:
            measures = list(part.getElementsByClass(stream.Measure))
            if measures:
                break
        if not measures:
            measures = list(score.recurse().getElementsByClass(stream.Measure))
        return len(measures)
    raise SymbolicAdapterError(f"unsupported symbolic score format: {suffix}")


def _build_candidates(
    *,
    corpus_root: Path,
    composers: set[str] | None,
    min_measures: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    canonical = corpus_root / "data" / "canonical"
    compositions = {row["composition_id"]: row for row in _read_csv(canonical / "compositions.csv")}
    works = {row["work_id"]: row for row in _read_csv(canonical / "works.csv")}
    scores = _read_csv(canonical / "score_versions.csv")
    rows: list[dict[str, object]] = []
    issues: list[dict[str, object]] = []
    for score in scores:
        work = works.get(_text(score.get("work_id")), {})
        composition = compositions.get(_text(work.get("composition_id")), {})
        composer = _composer(composition.get("composer"))
        if composers is not None and _composer_key(composer) not in composers:
            continue
        score_path = _text(score.get("source_file_path"))
        suffix = Path(score_path).suffix.casefold()
        issue_base = {
            "score_version_id": _text(score.get("score_version_id")),
            "work_id": _text(score.get("work_id")),
            "composer": composer,
            "score_path": score_path,
        }
        if suffix not in SUPPORTED_SCORE_EXTENSIONS:
            issues.append({**issue_base, "reason": "unsupported_score_format"})
            continue
        resolved = _resolve_asset(corpus_root, score_path)
        if not resolved.is_file():
            issues.append({**issue_base, "reason": "score_file_missing"})
            continue
        expected_hash = _text(score.get("sha256")).casefold()
        if len(expected_hash) == 64 and _sha256(resolved) != expected_hash:
            issues.append({**issue_base, "reason": "score_sha256_mismatch"})
            continue
        try:
            bars = _score_measure_count(resolved)
        except SymbolicAdapterError as exc:
            issues.append({**issue_base, "reason": str(exc)})
            continue
        if bars < min_measures:
            issues.append({**issue_base, "reason": f"too_few_measures:{bars}"})
            continue
        rows.append(
            {
                "composer": composer,
                "period": _period(composer),
                "composition_id": _text(work.get("composition_id")),
                "work_id": _text(score.get("work_id")),
                "movement_id": _text(score.get("work_id")),
                "score_path": score_path,
                "mxl_path": score_path if suffix in {".mxl", ".musicxml", ".xml"} else "",
                "midi_path": score_path if suffix in {".mid", ".midi"} else "",
                "bar_start": 1,
                "bar_end": bars,
                "score_format": suffix.lstrip("."),
                "rights_status": "licensed_for_research",
                "segment_type": "complete_movement",
                "fully_aligned": "false",
                "eligible_for_clamp": "true",
                "availability_status": "ready",
                "work_title": _text(work.get("title")) or _text(composition.get("title")),
                "score_version_id": _text(score.get("score_version_id")),
                "source_dataset": _text(score.get("source_dataset")),
                "source_record_ids": _text(score.get("source_record_ids")) or _text(score.get("source_record_id")),
                "score_priority": FORMAT_PRIORITY.get(suffix, 9),
            }
        )
    return rows, issues


def build_symbolic_adapter(
    *,
    corpus_root: str | Path,
    output_root: str | Path,
    composers: list[str],
    all_score_versions: bool = False,
    min_measures: int = 4,
) -> dict[str, object]:
    corpus = Path(corpus_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if output.exists():
        raise SymbolicAdapterError(f"output already exists; choose a new path: {output}")
    requested = {_composer_key(composer) for composer in composers} if composers else None
    candidates, issues = _build_candidates(
        corpus_root=corpus,
        composers=requested,
        min_measures=min_measures,
    )
    if not candidates:
        raise SymbolicAdapterError("no usable symbolic scores remain for the requested composers")
    if not all_score_versions:
        by_work: dict[str, dict[str, object]] = {}
        for row in sorted(
            candidates,
            key=lambda item: (
                str(item["composer"]),
                str(item["work_id"]),
                int(item["score_priority"]),
                str(item["source_dataset"]),
                str(item["score_path"]),
            ),
        ):
            by_work.setdefault(str(row["work_id"]), row)
        selected = list(by_work.values())
    else:
        selected = candidates
    selected.sort(key=lambda item: (str(item["composer"]), str(item["work_id"]), str(item["score_path"])))
    for row in selected:
        row["passage_id"] = _stable_id(_release(corpus), row["work_id"], row["score_version_id"])
        row.setdefault("recording_id", "")
        row.setdefault("audio_path", "")
        row.setdefault("alignment_path", "")
        row.setdefault("annotation_path", "")
        row.setdefault("alignment_format", "")
        row.setdefault("annotation_type", "")
    release = _release(corpus)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".piano-clamp-symbolic-", dir=output.parent) as temporary:
        stage = Path(temporary) / output.name
        stage.mkdir()
        manifest = stage / "piano_clamp_manifest.csv"
        issue_path = stage / "symbolic_import_issues.csv"
        extra_fields = (
            "composition_id",
            "work_title",
            "score_version_id",
            "source_dataset",
            "source_record_ids",
        )
        _write_csv(manifest, CORPUS_MANIFEST_FIELDS + extra_fields, selected)
        _write_csv(
            issue_path,
            ("score_version_id", "work_id", "composer", "score_path", "reason"),
            issues,
        )
        summary = {
            "adapter_schema": "piano-clamp-cpc-symbolic-adapter-1.0.0",
            "corpus_root": str(corpus),
            "corpus_release": release,
            "selected_rows": len(selected),
            "candidate_rows": len(candidates),
            "issue_rows": len(issues),
            "all_score_versions": all_score_versions,
            "min_measures": min_measures,
            "composer_counts": {},
            "source_dataset_counts": {},
            "manifest": str(output / manifest.name),
            "issues_manifest": str(output / issue_path.name),
        }
        for key in sorted({str(row["composer"]) for row in selected}):
            summary["composer_counts"][key] = sum(1 for row in selected if row["composer"] == key)
        for key in sorted({str(row["source_dataset"]) for row in selected}):
            summary["source_dataset_counts"][key] = sum(1 for row in selected if row["source_dataset"] == key)
        (stage / "adapter_metadata.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if len(read_corpus_manifest(manifest)) != len(selected):
            raise SymbolicAdapterError("generated manifest failed its row-count invariant")
        stage.replace(output)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--composer", action="append", dest="composers", required=True)
    parser.add_argument(
        "--all-score-versions",
        action="store_true",
        help="Keep every usable score version instead of selecting one preferred version per work.",
    )
    parser.add_argument("--min-measures", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_measures < 1:
        raise SystemExit("--min-measures must be positive")
    try:
        summary = build_symbolic_adapter(
            corpus_root=args.corpus_root,
            output_root=args.output_root,
            composers=args.composers,
            all_score_versions=args.all_score_versions,
            min_measures=args.min_measures,
        )
    except (OSError, SymbolicAdapterError) as exc:
        import traceback

        traceback.print_exc()
        raise SystemExit(f"error: {exc}\n{traceback.format_exc()}") from exc
    print(
        f"Prepared {summary['selected_rows']} symbolic rows from "
        f"{summary['candidate_rows']} usable candidates; excluded {summary['issue_rows']} rows."
    )
    print(f"Manifest: {summary['manifest']}")
    print(f"Issues: {summary['issues_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
