"""Prepare a deterministic PianoCoRe/PDMX score-passage bundle.

The source corpus is read-only.  This module reads selected MusicXML documents
directly from the pinned PianoCoRe raw-MIDI archive and writes a separate,
auditable bundle that satisfies :mod:`piano_clamp.data`'s manifest contract.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import re
import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from .data import MANIFEST_FIELDS, read_manifest, resolve_source_files


COMPOSERS = {
    "Chopin,_Frédéric": "Frédéric Chopin",
    "Mozart,_Wolfgang_Amadeus": "Wolfgang Amadeus Mozart",
}
DEFAULT_ARCHIVE = Path("data/raw/pianocore/1.0/PianoCoRe-1.0-raw-midi.zip")
DEFAULT_METADATA = Path("data/raw/pianocore/1.0/metadata.csv")
DEFAULT_FILE_MANIFEST = Path("manifests/pianocore-1.0-files.csv")
SOURCE_DATASET = "PDMX"
SELECTION_POLICY = (
    "PianoCoRe 1.0 tier A*, nonduplicate PDMX MusicXML; at least 16 measures; "
    "exclude horn-duo, two-piano, and song titles; deterministic diversity-first "
    "selection with at most two units per parent composition"
)
EXCLUDED_TITLE_PATTERNS = (
    "horn_duos",
    "2_pianos",
    "sehnsucht_nach_dem_frühling",
)

SELECTION_FIELDS = (
    "passage_id",
    "composer",
    "work_title",
    "movement",
    "pianocore_id",
    "score_dataset",
    "source_archive",
    "source_archive_member",
    "source_score_path",
    "source_score_sha256",
    "source_measure_count",
    "start_bar",
    "end_bar",
    "source_measure_start_label",
    "source_measure_end_label",
    "relative_path",
    "passage_sha256",
    "key",
    "meter",
    "selection_seed",
    "selection_policy",
)


class PreparationError(ValueError):
    """Raised when a reproducible passage bundle cannot be prepared."""


@dataclass(frozen=True)
class Candidate:
    """One unique PianoCoRe score candidate."""

    pianocore_id: str
    source_composer: str
    composer: str
    composition: str
    movement: str
    score_path: str
    measure_count: int = 0

    @property
    def identity(self) -> str:
        return "\x1f".join((self.source_composer, self.composition, self.movement, self.score_path))

    @property
    def work_title(self) -> str:
        composition = humanize(self.composition)
        movement = humanize(self.movement)
        return f"{composition}: {movement}" if movement else composition


def humanize(value: str) -> str:
    """Turn PianoCoRe path labels into readable manifest text."""

    return re.sub(r"\s+", " ", value.replace("_", " ")).strip()


def slug(value: str) -> str:
    value = value.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "-", value).strip("-") or "passage"


def local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def direct_children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if local_name(child) == name]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_archive_md5(corpus_root: Path) -> str:
    manifest = corpus_root / DEFAULT_FILE_MANIFEST
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["file_name"] == DEFAULT_ARCHIVE.name:
                if row["status"] != "verified_local":
                    raise PreparationError(f"PianoCoRe archive is not verified locally: {row['status']}")
                return row["md5"]
    raise PreparationError(f"archive is absent from pinned file manifest: {DEFAULT_ARCHIVE.name}")


def discover_candidates(corpus_root: Path) -> list[Candidate]:
    """Read the metadata gate and return unique PDMX score paths."""

    metadata = corpus_root / DEFAULT_METADATA
    unique: dict[tuple[str, str, str, str], Candidate] = {}
    with metadata.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            source_composer = row["composer"]
            if source_composer not in COMPOSERS:
                continue
            if row["score_dataset"] != SOURCE_DATASET or row["tier_a_star"] != "True":
                continue
            if row["is_duplicate"] == "True" or row["quality_label"] in {"low quality", "corrupted"}:
                continue
            if not row["score_xml_path"]:
                continue
            key = (
                source_composer,
                row["composition"],
                row["movement"],
                row["score_xml_path"],
            )
            candidate = Candidate(
                pianocore_id=row["id"],
                source_composer=source_composer,
                composer=COMPOSERS[source_composer],
                composition=row["composition"],
                movement=row["movement"],
                score_path=row["score_xml_path"],
            )
            prior = unique.get(key)
            if prior is None or candidate.pianocore_id < prior.pianocore_id:
                unique[key] = candidate
    if not unique:
        raise PreparationError("no PianoCoRe PDMX candidates matched the fixed selection gate")
    return sorted(unique.values(), key=lambda candidate: candidate.identity)


def mxl_musicxml(mxl_payload: bytes) -> bytes:
    """Return the root MusicXML document from an MXL byte string."""

    try:
        with zipfile.ZipFile(io.BytesIO(mxl_payload)) as archive:
            root_name = ""
            if "META-INF/container.xml" in archive.namelist():
                container = ET.fromstring(archive.read("META-INF/container.xml"))
                for element in container.iter():
                    if local_name(element) == "rootfile" and element.get("full-path"):
                        root_name = str(element.get("full-path"))
                        break
            if not root_name:
                root_name = next(
                    (
                        name
                        for name in archive.namelist()
                        if name.lower().endswith((".musicxml", ".xml"))
                        and not name.startswith("META-INF/")
                    ),
                    "",
                )
            if not root_name:
                raise PreparationError("MXL archive contains no MusicXML root document")
            return archive.read(root_name)
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise PreparationError(f"invalid MXL payload: {exc}") from exc


def score_parts(root: ET.Element) -> list[ET.Element]:
    if local_name(root) != "score-partwise":
        raise PreparationError(f"only score-partwise MusicXML is supported, found {local_name(root)!r}")
    parts = direct_children(root, "part")
    if not parts:
        raise PreparationError("MusicXML score contains no parts")
    return parts


def measure_count(xml_payload: bytes) -> int:
    try:
        root = ET.fromstring(xml_payload)
    except ET.ParseError as exc:
        raise PreparationError(f"invalid MusicXML: {exc}") from exc
    counts = [len(direct_children(part, "measure")) for part in score_parts(root)]
    if not counts or min(counts) == 0:
        raise PreparationError("MusicXML score contains a part with no measures")
    return min(counts)


def archive_member(candidate: Candidate) -> str:
    return f"PianoCoRe/raw/{candidate.score_path}"


def rank(seed: str, *values: str) -> str:
    payload = "\x1f".join((seed, *values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def excluded_title(candidate: Candidate) -> bool:
    normalized = candidate.composition.casefold()
    return any(pattern in normalized for pattern in EXCLUDED_TITLE_PATTERNS)


def diversity_order(
    candidates: Iterable[Candidate], *, seed: str, max_per_composition: int
) -> list[Candidate]:
    """Order candidates by parent-composition rounds, then a fixed hash."""

    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.composition].append(candidate)
    for composition in grouped:
        grouped[composition].sort(key=lambda item: rank(seed, item.identity))
    compositions = sorted(grouped, key=lambda value: rank(seed, value))
    ordered: list[Candidate] = []
    for round_index in range(max_per_composition):
        for composition in compositions:
            choices = grouped[composition]
            if round_index < len(choices):
                ordered.append(choices[round_index])
    return ordered


def select_balanced(
    candidates: Iterable[Candidate],
    *,
    target_per_composer: int,
    window_bars: int,
    seed: str,
    max_per_composition: int,
) -> list[Candidate]:
    by_composer: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.measure_count >= window_bars and not excluded_title(candidate):
            by_composer[candidate.composer].append(candidate)
    missing = sorted(set(COMPOSERS.values()) - set(by_composer))
    if missing:
        raise PreparationError(f"no eligible candidates for: {', '.join(missing)}")
    available = {
        composer: diversity_order(
            rows, seed=seed, max_per_composition=max_per_composition
        )
        for composer, rows in by_composer.items()
    }
    sample_size = min(target_per_composer, *(len(rows) for rows in available.values()))
    if sample_size < 2:
        raise PreparationError("selection leaves fewer than two works per composer")
    selected = [candidate for composer in sorted(available) for candidate in available[composer][:sample_size]]
    return sorted(selected, key=lambda candidate: (candidate.composer, candidate.work_title))


def _attribute_key(element: ET.Element) -> tuple[str, str]:
    return local_name(element), element.get("number", "")


def _carry_effective_attributes(measures: list[ET.Element], start_index: int) -> None:
    """Make the first retained measure self-contained after cropping."""

    effective: dict[tuple[str, str], ET.Element] = {}
    first_attributes: ET.Element | None = None
    for measure_index, measure in enumerate(measures[: start_index + 1]):
        for attributes in direct_children(measure, "attributes"):
            if measure_index == start_index and first_attributes is None:
                first_attributes = attributes
            for child in list(attributes):
                effective[_attribute_key(child)] = copy.deepcopy(child)
    first_measure = measures[start_index]
    if first_attributes is None:
        namespace = first_measure.tag[: first_measure.tag.index("}") + 1] if "}" in first_measure.tag else ""
        first_attributes = ET.Element(f"{namespace}attributes")
        first_measure.insert(0, first_attributes)
    else:
        for child in list(first_attributes):
            first_attributes.remove(child)
    for child in effective.values():
        first_attributes.append(child)


def crop_musicxml(xml_payload: bytes, *, start_index: int, window_bars: int) -> tuple[bytes, str, str]:
    """Crop every part by positional bar index and return XML plus source labels."""

    try:
        root = ET.fromstring(xml_payload)
    except ET.ParseError as exc:
        raise PreparationError(f"invalid MusicXML: {exc}") from exc
    parts = score_parts(root)
    measures_by_part = [direct_children(part, "measure") for part in parts]
    available = min(len(measures) for measures in measures_by_part)
    end_index = start_index + window_bars
    if start_index < 0 or end_index > available:
        raise PreparationError(
            f"requested positional measures {start_index + 1}-{end_index}, score has {available}"
        )
    source_start = measures_by_part[0][start_index].get("number", str(start_index + 1))
    source_end = measures_by_part[0][end_index - 1].get("number", str(end_index))
    for part, measures in zip(parts, measures_by_part):
        _carry_effective_attributes(measures, start_index)
        retained = set(measures[start_index:end_index])
        for measure in measures:
            if measure not in retained:
                part.remove(measure)
    if "}" in root.tag:
        ET.register_namespace("", root.tag[1 : root.tag.index("}")])
    buffer = io.BytesIO()
    ET.ElementTree(root).write(buffer, encoding="utf-8", xml_declaration=True)
    cropped = buffer.getvalue()
    if measure_count(cropped) != window_bars:
        raise PreparationError("cropped MusicXML failed its measure-count invariant")
    return cropped, str(source_start), str(source_end)


def first_attributes(root: ET.Element) -> ET.Element | None:
    for part in score_parts(root):
        for measure in direct_children(part, "measure"):
            attributes = direct_children(measure, "attributes")
            if attributes:
                return attributes[0]
    return None


def child_text(element: ET.Element | None, name: str) -> str:
    if element is None:
        return ""
    for child in element.iter():
        if local_name(child) == name:
            return (child.text or "").strip()
    return ""


def key_label(root: ET.Element) -> str:
    attributes = first_attributes(root)
    key = next(iter(direct_children(attributes, "key")), None) if attributes is not None else None
    fifths_text = child_text(key, "fifths")
    if not fifths_text:
        return "unspecified key signature"
    fifths = int(fifths_text)
    major = {
        -7: "C-flat", -6: "G-flat", -5: "D-flat", -4: "A-flat", -3: "E-flat",
        -2: "B-flat", -1: "F", 0: "C", 1: "G", 2: "D", 3: "A", 4: "E",
        5: "B", 6: "F-sharp", 7: "C-sharp",
    }.get(fifths, "unknown")
    minor = {
        -7: "A-flat", -6: "E-flat", -5: "B-flat", -4: "F", -3: "C", -2: "G",
        -1: "D", 0: "A", 1: "E", 2: "B", 3: "F-sharp", 4: "C-sharp",
        5: "G-sharp", 6: "D-sharp", 7: "A-sharp",
    }.get(fifths, "unknown")
    mode = child_text(key, "mode")
    if mode == "major":
        return f"{major} major"
    if mode == "minor":
        return f"{minor} minor"
    return f"{major} major / {minor} minor key signature"


def meter_label(root: ET.Element) -> str:
    attributes = first_attributes(root)
    time = next(iter(direct_children(attributes, "time")), None) if attributes is not None else None
    beats = child_text(time, "beats")
    beat_type = child_text(time, "beat-type")
    return f"{beats}/{beat_type}" if beats and beat_type else "unspecified meter"


def passage_id(candidate: Candidate, start_bar: int, end_bar: int) -> str:
    prefix = "chopin" if candidate.composer == "Frédéric Chopin" else "mozart"
    identity_hash = hashlib.sha256(candidate.identity.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_pdmx_{slug(candidate.composition)[:28]}_{identity_hash}_b{start_bar:03d}_{end_bar:03d}"


def write_csv(path: Path, fields: Iterable[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def build_bundle(
    *,
    corpus_root: Path,
    output_root: Path,
    target_per_composer: int = 16,
    window_bars: int = 16,
    seed: str = "20260819",
    max_per_composition: int = 2,
) -> Path:
    """Build and validate one immutable external passage-data bundle."""

    corpus_root = corpus_root.resolve()
    output_root = output_root.resolve()
    archive_path = corpus_root / DEFAULT_ARCHIVE
    if not archive_path.is_file():
        raise PreparationError(f"PianoCoRe archive does not exist: {archive_path}")
    if output_root.exists():
        raise PreparationError(f"output already exists; choose a new versioned path: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    expected_md5 = pinned_archive_md5(corpus_root)
    actual_md5 = digest_file(archive_path, "md5")
    if actual_md5 != expected_md5:
        raise PreparationError(
            f"PianoCoRe archive MD5 mismatch: expected {expected_md5}, found {actual_md5}"
        )

    candidates = discover_candidates(corpus_root)
    inspected: list[Candidate] = []
    with zipfile.ZipFile(archive_path) as outer:
        for candidate in candidates:
            member = archive_member(candidate)
            try:
                xml_payload = mxl_musicxml(outer.read(member))
            except KeyError as exc:
                raise PreparationError(f"archive member is missing: {member}") from exc
            inspected.append(replace(candidate, measure_count=measure_count(xml_payload)))
        selected = select_balanced(
            inspected,
            target_per_composer=target_per_composer,
            window_bars=window_bars,
            seed=seed,
            max_per_composition=max_per_composition,
        )

        with tempfile.TemporaryDirectory(prefix=".piano-clamp-prepare-", dir=output_root.parent) as temporary:
            stage = Path(temporary) / output_root.name
            (stage / "manifests").mkdir(parents=True)
            manifest_rows: list[dict[str, object]] = []
            selection_rows: list[dict[str, object]] = []
            for candidate in selected:
                member = archive_member(candidate)
                source_mxl = outer.read(member)
                xml_payload = mxl_musicxml(source_mxl)
                start_bar = (candidate.measure_count - window_bars) // 2 + 1
                end_bar = start_bar + window_bars - 1
                cropped, source_start, source_end = crop_musicxml(
                    xml_payload,
                    start_index=start_bar - 1,
                    window_bars=window_bars,
                )
                root = ET.fromstring(cropped)
                item_id = passage_id(candidate, start_bar, end_bar)
                composer_dir = "chopin" if candidate.composer == "Frédéric Chopin" else "mozart"
                relative_path = Path("scores") / composer_dir / f"{item_id}.musicxml"
                destination = stage / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(cropped)
                movement = humanize(candidate.movement) or "complete score"
                key = key_label(root)
                meter = meter_label(root)
                manifest_rows.append(
                    {
                        "passage_id": item_id,
                        "composer": candidate.composer,
                        "work_title": candidate.work_title,
                        "movement": movement,
                        "relative_path": relative_path.as_posix(),
                        "start_bar": start_bar,
                        "end_bar": end_bar,
                        "key": key,
                        "meter": meter,
                    }
                )
                selection_rows.append(
                    {
                        **manifest_rows[-1],
                        "pianocore_id": candidate.pianocore_id,
                        "score_dataset": SOURCE_DATASET,
                        "source_archive": DEFAULT_ARCHIVE.as_posix(),
                        "source_archive_member": member,
                        "source_score_path": candidate.score_path,
                        "source_score_sha256": sha256_bytes(source_mxl),
                        "source_measure_count": candidate.measure_count,
                        "source_measure_start_label": source_start,
                        "source_measure_end_label": source_end,
                        "passage_sha256": sha256_bytes(cropped),
                        "selection_seed": seed,
                        "selection_policy": SELECTION_POLICY,
                    }
                )

            manifest_rows.sort(key=lambda row: str(row["passage_id"]))
            selection_rows.sort(key=lambda row: str(row["passage_id"]))
            manifest_path = stage / "manifests/chopin_mozart.csv"
            write_csv(manifest_path, MANIFEST_FIELDS, manifest_rows)
            write_csv(stage / "manifests/selection.csv", SELECTION_FIELDS, selection_rows)
            normalized = read_manifest(manifest_path)
            resolved = resolve_source_files(normalized, stage)
            if len(resolved) != len(selected):
                raise PreparationError("prepared manifest did not resolve every selected passage")
            counts: dict[str, int] = defaultdict(int)
            for row in manifest_rows:
                counts[str(row["composer"])] += 1
            provenance = {
                "bundle_schema": "1.0.0",
                "source_corpus": corpus_root.as_posix(),
                "source_corpus_release": "cpc-2026-08-19.2",
                "pianocore_release": "1.0",
                "pianocore_archive": DEFAULT_ARCHIVE.as_posix(),
                "pianocore_archive_md5": actual_md5,
                "score_dataset": SOURCE_DATASET,
                "window_bars": window_bars,
                "target_per_composer": target_per_composer,
                "actual_by_composer": dict(sorted(counts.items())),
                "selection_seed": seed,
                "max_units_per_parent_composition": max_per_composition,
                "selection_policy": SELECTION_POLICY,
            }
            (stage / "provenance.json").write_text(
                json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (stage / "README.md").write_text(
                "# Chopin–Mozart PDMX passage bundle\n\n"
                "This immutable analysis input was generated from the verified PianoCoRe 1.0 "
                "raw-MIDI archive. It contains one centered 16-bar MusicXML passage from each "
                "selected work-unit and an exactly balanced composer sample.\n\n"
                "Use this directory as `MUSIC_DATA_ROOT`. The consumer manifest is "
                "`manifests/chopin_mozart.csv`; row-level source hashes and original archive "
                "members are in `manifests/selection.csv`. See `provenance.json` for the fixed "
                "selection policy. These passages remain subject to PianoCoRe/PDMX source terms.\n",
                encoding="utf-8",
            )
            stage.replace(output_root)
    return output_root / "manifests/chopin_mozart.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--target-per-composer", type=int, default=16)
    parser.add_argument("--window-bars", type=int, default=16)
    parser.add_argument("--seed", default="20260819")
    parser.add_argument("--max-per-composition", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = build_bundle(
            corpus_root=args.corpus_root,
            output_root=args.output_root,
            target_per_composer=args.target_per_composer,
            window_bars=args.window_bars,
            seed=args.seed,
            max_per_composition=args.max_per_composition,
        )
    except (OSError, csv.Error, zipfile.BadZipFile, PreparationError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
