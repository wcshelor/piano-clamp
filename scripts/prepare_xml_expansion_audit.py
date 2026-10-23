#!/usr/bin/env python3
"""Metadata-only XML expansion audit with explicit validation separation.

Default is metadata-only audit (no HPC files required). Asset validation requires
an explicit --corpus-root (supports ${PIANO_CLAMP_CORPUS_ROOT} expansion).

Outputs (default to reports/hpc/xml_expansion_audit/review):
  - decision.csv
  - alias_mapping.csv
  - conflict_report.json
  - summary.json
  - expanded_manifest.csv  (candidate, availability_status=needs_validation)
  - expanded_manifest.validated.csv + validation_certificate.json (only after successful validation)

Do not modify canonical source CSVs or existing manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from piano_clamp.data_loading import CORPUS_MANIFEST_FIELDS
from piano_clamp.xml_expansion import (
    VALIDATOR_VERSION,
    derive_candidate_identity,
    detect_group_conflicts,
    find_score_version_for_selection,
    is_uncertain_instrumentation,
    normalize_composer,
    stable_analysis_work_id,
    validate_override,
    validate_representative_override,
)

_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def _text(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_atomic(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # ensure temp in same dir for atomic rename
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _preflight(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"error: {label} does not exist: {path}")
    if not path.stat().st_size:
        raise SystemExit(f"error: {label} is empty: {path}")


def _resolve_corpus_root(value: str | Path | None, environ: dict[str, str] | None = None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    env = environ if environ is not None else os.environ
    m = _ENV_REF.fullmatch(text)
    if m:
        var = m.group(1)
        resolved = env.get(var)
        if not resolved:
            hint = (
                "On the HPC: export PIANO_CLAMP_CORPUS_ROOT=/home/student/w/wshelor/share/datasets/classical-performance-corpus"
                if var == "PIANO_CLAMP_CORPUS_ROOT"
                else f"Export {var} before running"
            )
            raise SystemExit(f"error: corpus_root references {var}, but that environment variable is not set. {hint}.")
        text = resolved
    else:
        # also support generic ${VAR} inside string via expandvars check
        expanded = os.path.expandvars(text)
        if "$" in expanded:
            raise SystemExit(f"error: corpus_root contains an unresolved environment variable: {text}")
        text = expanded
    p = Path(text).expanduser()
    if not p.is_absolute():
        # Resolve relative to repo root for convenience
        p = (ROOT / p).resolve()
    else:
        p = p.resolve()
    if not p.is_dir():
        raise SystemExit(f"error: corpus_root does not exist or is not a directory: {p}")
    if not os.access(p, os.R_OK | os.X_OK):
        raise SystemExit(f"error: corpus_root is not readable: {p}")
    return p


def _load_overrides(path: Path | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    if not path.is_file():
        raise SystemExit(f"error: overrides file does not exist: {path}")
    rows = _read_csv(path)
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        sha = _text(r.get("sha256")).casefold()
        if not sha:
            raise SystemExit(f"error: override row missing sha256: {r}")
        if sha in out:
            raise SystemExit(f"error: duplicate override for sha {sha}")
        if not _text(r.get("reason")):
            raise SystemExit(f"error: override for {sha} missing reason")
        if not _text(r.get("evidence")):
            raise SystemExit(f"error: override for {sha} missing evidence")
        out[sha] = {k: _text(v) for k, v in r.items()}
    return out


def _load_representative_overrides(
    path: Path | None,
    identity_to_shas: dict[str, list[str]],
    analysis_id_to_key: dict[str, str],
) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    if not path.is_file():
        raise SystemExit(f"error: representative overrides file does not exist: {path}")
    rows = _read_csv(path)
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        aid = _text(r.get("analysis_work_id"))
        sha = _text(r.get("representative_sha256")).casefold()
        if not aid:
            raise SystemExit(f"error: representative override row missing analysis_work_id: {r}")
        if not sha:
            raise SystemExit(f"error: representative override for {aid} missing representative_sha256")
        if aid in out:
            raise SystemExit(f"error: duplicate representative override for analysis_work_id {aid}")
        if not _text(r.get("reason")):
            raise SystemExit(f"error: representative override for {aid} missing reason")
        if not _text(r.get("evidence")):
            raise SystemExit(f"error: representative override for {aid} missing evidence")
        # validate against groups
        try:
            validate_representative_override(r, identity_to_shas, analysis_id_to_key)
        except ValueError as exc:
            raise SystemExit(f"error: invalid representative override for {aid}: {exc}")
        out[aid] = {k: _text(v) for k, v in r.items()}
    return out


def _composer_period(composer: str) -> str:
    key = normalize_composer(composer)
    if key == "Chopin":
        return "Romantic"
    if key == "Mozart":
        return "Classical"
    if key == "Bach":
        return "Baroque"
    return ""


def _relative_score_path(xml_path: str) -> str:
    hpc_prefix = "/home/student/w/wshelor/share/datasets/classical-performance-corpus/"
    share_prefix = "/share/users/student/w/wshelor/datasets/classical-performance-corpus/"
    if xml_path.startswith(hpc_prefix):
        rel = xml_path[len(hpc_prefix) :]
    elif xml_path.startswith(share_prefix):
        rel = xml_path[len(share_prefix) :]
    else:
        rel = xml_path.lstrip("/")
    return rel.lstrip("/")


def _validate_score_file(
    corpus_root: Path,
    relative_path: str,
    expected_sha: str,
    bar_end: int,
) -> list[str]:
    """Return list of error messages for a single score file."""
    errors: list[str] = []
    # 1. Path resolves safely within corpus root
    try:
        resolved = _resolve_score_path_safe(corpus_root, relative_path)
    except ValueError as exc:
        return [f"path escapes corpus_root: {exc}"]
    # 2. File exists and is readable
    if not resolved.is_file():
        return [f"file does not exist: {resolved} (relative {relative_path!r} under corpus_root {corpus_root})"]
    if not os.access(resolved, os.R_OK):
        return [f"file is not readable: {resolved}"]
    # 3. Supported format
    suffix = resolved.suffix.lower()
    if suffix not in {".mxl", ".musicxml", ".xml"}:
        return [f"unsupported score format {suffix!r} for {resolved}; expected .mxl, .musicxml, or .xml"]
    # 4. SHA matches
    try:
        actual_sha = _hash_file(resolved)
    except OSError as exc:
        return [f"cannot read file for checksum: {exc}"]
    if actual_sha.casefold() != expected_sha.casefold():
        return [f"SHA-256 mismatch for {resolved}: expected {expected_sha}, found {actual_sha}"]
    # 5. MXL container / XML parses
    try:
        # Use existing helpers: for .mxl, extract via zip; for .xml/.musicxml, read directly
        xml_bytes: bytes
        if suffix == ".mxl":
            # Reuse logic from prepare/features: read container
            with zipfile.ZipFile(resolved) as archive:
                names = set(archive.namelist())
                root_name = None
                if "META-INF/container.xml" in names:
                    container = ET.fromstring(archive.read("META-INF/container.xml"))
                    for el in container.iter():
                        tag = el.tag.rsplit("}", 1)[-1]
                        if tag == "rootfile" and el.get("full-path") in names:
                            root_name = el.get("full-path")
                            break
                if root_name is None:
                    root_name = next(
                        (n for n in archive.namelist() if n.lower().endswith((".musicxml", ".xml")) and not n.startswith("META-INF/")),
                        None,
                    )
                if root_name is None:
                    return [f"MXL archive contains no MusicXML root document: {resolved}"]
                xml_bytes = archive.read(root_name)
        else:
            xml_bytes = resolved.read_bytes()
        root = ET.fromstring(xml_bytes)
    except (zipfile.BadZipFile, ET.ParseError, OSError, KeyError) as exc:
        return [f"invalid MXL/XML container for {resolved}: {exc}"]
    # Check root tag
    tag_local = root.tag.rsplit("}", 1)[-1]
    if tag_local != "score-partwise":
        return [f"unsupported MusicXML root {tag_local!r} for {resolved}; only score-partwise is supported"]
    # 6. Pitched content and measure coverage
    # Count measures per part and pitched notes
    try:
        # Use helpers similar to prepare.measure_count / features
        parts = [c for c in list(root) if c.tag.rsplit("}", 1)[-1] == "part"]
        if not parts:
            return [f"MusicXML score contains no parts: {resolved}"]
        # pitched notes check: look for note/pitch
        pitched = 0
        part_measure_counts: list[int] = []
        for part in parts:
            measures = [c for c in list(part) if c.tag.rsplit("}", 1)[-1] == "measure"]
            part_measure_counts.append(len(measures))
            for meas in measures:
                for note in (c for c in list(meas) if c.tag.rsplit("}", 1)[-1] == "note"):
                    has_pitch = any(ch.tag.rsplit("}", 1)[-1] == "pitch" for ch in list(note))
                    if has_pitch:
                        # Check not rest
                        is_rest = any(ch.tag.rsplit("}", 1)[-1] == "rest" for ch in list(note))
                        if not is_rest:
                            pitched += 1
        if pitched == 0:
            return [f"score has no pitched notes: {resolved}"]
        min_measures = min(part_measure_counts) if part_measure_counts else 0
        if min_measures == 0:
            return [f"score contains a part with no measures: {resolved}"]
        if bar_end > min_measures:
            return [f"measure range 1-{bar_end} exceeds shortest part measure count {min_measures} for {resolved}"]
    except Exception as exc:
        return [f"failed to validate measure coverage for {resolved}: {exc}"]
    return []


def _resolve_score_path_safe(corpus_root: Path, relative: str) -> Path:
    root = corpus_root.resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"relative path must be within corpus_root without traversal: {relative!r}")
    candidate = (root / rel).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes corpus_root: {relative!r} -> {candidate}")
    return candidate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usable-csv", type=Path, default=ROOT / "reports/hpc/xml_expansion_audit/input/usable_xml_scores.csv")
    parser.add_argument("--works-csv", type=Path, default=ROOT / "reports/hpc/xml_expansion_audit/input/works.csv")
    parser.add_argument("--compositions-csv", type=Path, default=ROOT / "reports/hpc/xml_expansion_audit/input/compositions.csv")
    parser.add_argument("--score-versions-csv", type=Path, default=ROOT / "reports/hpc/xml_expansion_audit/input/score_versions.csv")
    parser.add_argument("--current-main-manifest", type=Path, default=ROOT / "reports/hpc/xml_expansion_audit/input/current_main_manifest.csv")
    parser.add_argument("--current-bach-manifest", type=Path, default=ROOT / "reports/hpc/xml_expansion_audit/input/current_bach_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/hpc/xml_expansion_audit/review")
    parser.add_argument("--overrides-csv", type=Path, default=None, help="CSV with sha256, resolved_catalogue, resolved_movement, reason, evidence")
    parser.add_argument(
        "--representative-overrides",
        type=Path,
        default=None,
        help="CSV with analysis_work_id, representative_sha256, reason, evidence for edition selection",
    )
    parser.add_argument("--corpus-root", type=str, default=None, help="Explicit corpus root for asset validation (supports ${PIANO_CLAMP_CORPUS_ROOT})")
    parser.add_argument("--revalidate", action="store_true", help="Legacy flag: requires --corpus-root and performs full asset validation")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args(argv)

    # Resolve corpus root if provided (including legacy --revalidate without explicit root should error)
    corpus_root: Path | None = None
    if args.corpus_root is not None:
        try:
            corpus_root = _resolve_corpus_root(args.corpus_root)
        except SystemExit as exc:
            raise
    if args.revalidate and corpus_root is None:
        raise SystemExit(
            "error: --revalidate requires --corpus-root with explicit corpus path (e.g., --corpus-root ${PIANO_CLAMP_CORPUS_ROOT}). "
            "On the HPC: export PIANO_CLAMP_CORPUS_ROOT=/home/student/w/wshelor/share/datasets/classical-performance-corpus"
        )

    # Preflight inputs
    for p, label in [
        (args.usable_csv, "usable_xml_scores.csv"),
        (args.works_csv, "works.csv"),
        (args.compositions_csv, "compositions.csv"),
        (args.score_versions_csv, "score_versions.csv"),
        (args.current_main_manifest, "current_main_manifest.csv"),
        (args.current_bach_manifest, "current_bach_manifest.csv"),
    ]:
        _preflight(p, label)

    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise SystemExit(f"error: output-dir exists and is not a directory: {args.output_dir}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.force:
        raise SystemExit(f"error: output directory already exists and is not empty: {args.output_dir} (use --force to overwrite)")

    overrides = _load_overrides(args.overrides_csv)

    usable = _read_csv(args.usable_csv)
    works = {r["work_id"]: r for r in _read_csv(args.works_csv)}
    comps = {r["composition_id"]: r for r in _read_csv(args.compositions_csv)}
    score_versions = {r["score_version_id"]: r for r in _read_csv(args.score_versions_csv)}
    main_rows = _read_csv(args.current_main_manifest)
    bach_rows = _read_csv(args.current_bach_manifest)

    # Group by sha256 preserving every source ID/title/path
    hash_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in usable:
        sha = _text(r.get("sha256")).casefold()
        if not sha:
            raise SystemExit(f"error: usable row missing sha256: {r}")
        hash_groups[sha].append(r)

    # Derive identities for selected manifests to compare
    selected_identities: set[str] = set()
    selected_identity_details: dict[str, list[str]] = defaultdict(list)
    selected_work_ids: set[str] = set()
    for manifest_row in main_rows + bach_rows:
        selected_work_ids.add(_text(manifest_row.get("work_id")))
        comp_id = _text(manifest_row.get("composition_id"))
        composer = _text(manifest_row.get("composer"))
        synth = {
            "composer": composer,
            "title": _text(manifest_row.get("work_title")),
            "xml_path": _text(manifest_row.get("score_path")) or _text(manifest_row.get("mxl_path")),
            "composition_id": comp_id,
        }
        cat, mov, _ = derive_candidate_identity(synth, comps)
        if cat or mov:
            key = f"{normalize_composer(composer).casefold()}|{(cat or '').casefold()}|{(mov or '').casefold()}"
            selected_identities.add(key)
            selected_identity_details[key].append(_text(manifest_row.get("work_id")))

    # Process each hash group: first pass derive identities
    decisions: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    identity_to_shas: dict[str, list[str]] = defaultdict(list)
    sha_to_identity: dict[str, tuple[str | None, str | None]] = {}
    sha_to_analysis_id: dict[str, str] = {}

    for sha, rows in sorted(hash_groups.items()):
        conflict_info = detect_group_conflicts(rows, works, comps)
        has_override = sha in overrides
        override = overrides.get(sha)
        if has_override:
            try:
                validate_override(override, rows, works, comps)
            except ValueError as exc:
                raise SystemExit(f"error: invalid override for sha {sha}: {exc}")

        quarantine_reason = None
        if conflict_info["conflicting_catalogues"]:
            quarantine_reason = f"conflicting_catalogues:{conflict_info['catalogue_set']}"
        elif conflict_info["conflicting_movements"]:
            quarantine_reason = f"conflicting_movements:{conflict_info['movement_set']}"
        elif conflict_info["title_path_catalogue_mismatches"]:
            quarantine_reason = f"title_path_catalogue_mismatch rows {conflict_info['title_path_catalogue_mismatches']}"
        elif conflict_info["title_path_movement_mismatches"]:
            quarantine_reason = f"title_path_movement_mismatch rows {conflict_info['title_path_movement_mismatches']}"

        if quarantine_reason and has_override:
            resolved_cat = _text(override.get("resolved_catalogue"))
            resolved_mov = _text(override.get("resolved_movement"))
            composer = _text(rows[0].get("composer"))
            cat = resolved_cat or None
            mov = resolved_mov or None
            from piano_clamp.xml_expansion import normalize_k_token, normalize_movement

            if cat:
                cat = normalize_k_token(cat)
            if mov:
                mov = normalize_movement(mov)
            analysis_id = stable_analysis_work_id(composer, cat, mov)
            sha_to_identity[sha] = (cat, mov)
            sha_to_analysis_id[sha] = analysis_id
            identity_key = f"{normalize_composer(composer).casefold()}|{(cat or '').casefold()}|{(mov or '').casefold()}"
            identity_to_shas[identity_key].append(sha)
            conflicts.append(
                {
                    "sha256": sha,
                    "type": "resolved_by_override",
                    "original_conflict": quarantine_reason,
                    "resolved_catalogue": cat,
                    "resolved_movement": mov,
                    "reason": _text(override.get("reason")),
                    "evidence": conflict_info,
                    "work_ids": sorted({r["work_id"] for r in rows}),
                    "titles": sorted({r["title"] for r in rows}),
                    "xml_paths": sorted({r["xml_path"] for r in rows}),
                }
            )
            quarantine_reason = None
        elif quarantine_reason:
            conflicts.append(
                {
                    "sha256": sha,
                    "type": "quarantined_conflict",
                    "reason": quarantine_reason,
                    "evidence": conflict_info,
                    "work_ids": sorted({r["work_id"] for r in rows}),
                    "titles": sorted({r["title"] for r in rows}),
                    "xml_paths": sorted({r["xml_path"] for r in rows}),
                }
            )

        cat = None
        mov = None
        analysis_id = None
        missing_evidence = conflict_info["missing_evidence"]
        uncertain = False
        for r in rows:
            if is_uncertain_instrumentation(_text(r.get("title"))):
                uncertain = True
                break

        if not quarantine_reason:
            cat = None
            mov = None
            evidence = ""
            chosen_row = rows[0]
            for cand in sorted(rows, key=lambda r: (0 if derive_candidate_identity(r, comps)[0] else 1, r["work_id"])):
                c, m, ev = derive_candidate_identity(cand, comps)
                if c:
                    cat, mov, evidence = c, m, ev
                    chosen_row = cand
                    break
            else:
                cat, mov, evidence = derive_candidate_identity(rows[0], comps)
                chosen_row = rows[0]
            composer_key = normalize_composer(chosen_row.get("composer"))
            if not cat and composer_key in ("Mozart", "Chopin", "Bach"):
                missing_evidence = True
            if not cat and not mov:
                missing_evidence = True
            if missing_evidence or conflict_info["missing_evidence"]:
                quarantine_reason = "missing_identity_evidence"
                conflicts.append(
                    {
                        "sha256": sha,
                        "type": "missing_evidence",
                        "evidence": conflict_info,
                        "work_ids": sorted({r["work_id"] for r in rows}),
                    }
                )
            else:
                analysis_id = stable_analysis_work_id(_text(chosen_row.get("composer")), cat, mov)
                sha_to_identity[sha] = (cat, mov)
                sha_to_analysis_id[sha] = analysis_id
                identity_key = f"{normalize_composer(_text(chosen_row.get('composer'))).casefold()}|{(cat or '').casefold()}|{(mov or '').casefold()}"
                identity_to_shas[identity_key].append(sha)

        composer = _text(rows[0].get("composer"))
        for r in rows:
            alias_rows.append(
                {
                    "sha256": sha,
                    "work_id": _text(r.get("work_id")),
                    "composition_id": _text(r.get("composition_id")),
                    "title": _text(r.get("title")),
                    "xml_path": _text(r.get("xml_path")),
                    "analysis_work_id": analysis_id or "",
                    "candidate_catalogue": cat or "",
                    "candidate_movement": mov or "",
                }
            )

        has_parent_selected = any(_text(r.get("parent_composition_already_selected")).casefold() == "true" for r in rows)
        decisions.append(
            {
                "sha256": sha,
                "composer": composer,
                "record_count": len(rows),
                "work_ids": ";".join(sorted({r["work_id"] for r in rows})),
                "titles": ";".join(sorted({r["title"] for r in rows})),
                "xml_paths": ";".join(sorted({r["xml_path"] for r in rows})),
                "candidate_catalogue": cat or "",
                "candidate_movement": mov or "",
                "analysis_work_id": analysis_id or "",
                "quarantine_reason": quarantine_reason or "",
                "uncertain_instrumentation": str(uncertain),
                "status": _text(rows[0].get("status")),
                "_cat": cat,
                "_mov": mov,
                "_composer": composer,
                "_quarantine": quarantine_reason,
                "_uncertain": uncertain,
                "_analysis_id": analysis_id,
                "_parent_selected": has_parent_selected,
                "_rows": rows,
            }
        )

    # Build mapping for representative validation: analysis_id -> identity_key
    analysis_id_to_key: dict[str, str] = {}
    for dec in decisions:
        aid = dec["_analysis_id"]
        if aid:
            cat = dec["_cat"]
            mov = dec["_mov"]
            composer = dec["_composer"]
            key = f"{normalize_composer(composer).casefold()}|{(cat or '').casefold()}|{(mov or '').casefold()}"
            analysis_id_to_key[aid] = key

    # Load representative overrides now that identity_to_shas is populated
    rep_overrides = _load_representative_overrides(args.representative_overrides, identity_to_shas, analysis_id_to_key)

    # Second pass: handle deduplication and already_represented
    for dec in decisions:
        group_work_ids = set(dec["work_ids"].split(";")) if dec["work_ids"] else set()
        if group_work_ids & selected_work_ids:
            dec["decision"] = "already_represented"
            dec["decision_reason"] = f"already represented via work_id overlap {sorted(group_work_ids & selected_work_ids)}"
            dec["quarantine_reason"] = dec["quarantine_reason"] or ""
            continue
        if dec.get("_parent_selected"):
            dec["decision"] = "already_represented"
            dec["decision_reason"] = "parent_composition_already_selected"
            dec["quarantine_reason"] = dec["quarantine_reason"] or ""
            continue
        if not dec["_analysis_id"]:
            continue
        cat = dec["_cat"]
        mov = dec["_mov"]
        composer = dec["_composer"]
        key = f"{normalize_composer(composer).casefold()}|{(cat or '').casefold()}|{(mov or '').casefold()}"
        if key in selected_identities:
            dec["decision"] = "already_represented"
            dec["decision_reason"] = f"already represented via {selected_identity_details[key]}"
            dec["quarantine_reason"] = dec["quarantine_reason"] or ""
            continue
        if dec["_quarantine"]:
            dec["decision"] = "quarantined"
            dec["decision_reason"] = dec["_quarantine"]
            continue
        if dec["_uncertain"]:
            dec["decision"] = "unresolved_uncertain_instrumentation"
            dec["decision_reason"] = "uncertain instrumentation"
            continue
        if not dec["_analysis_id"]:
            dec["decision"] = "unresolved_missing_evidence"
            dec["decision_reason"] = "missing catalogue/movement"
            continue
        shas_for_identity = identity_to_shas.get(key, [])
        if len(shas_for_identity) > 1:
            # Require explicit representative selection
            aid = dec["_analysis_id"]
            chosen = rep_overrides.get(aid)
            if chosen is None:
                dec["decision"] = "unresolved_ambiguous_edition"
                dec["decision_reason"] = f"ambiguous: multiple editions share candidate identity {aid} ({sorted(shas_for_identity)}); requires explicit representative selection"
                continue
            chosen_sha = _text(chosen.get("representative_sha256")).casefold()
            if dec["sha256"].casefold() == chosen_sha:
                dec["decision"] = "approved"
                dec["decision_reason"] = f"representative for {aid} via override: {_text(chosen.get('reason'))}"
            else:
                dec["decision"] = "excluded_not_selected"
                dec["decision_reason"] = f"excluded: not selected as representative for {aid}; chosen is {chosen_sha}"
            continue
        dec["decision"] = "approved"
        dec["decision_reason"] = "passes all checks"

    for dec in decisions:
        if "decision" not in dec:
            if dec["_quarantine"]:
                dec["decision"] = "quarantined"
                dec["decision_reason"] = dec["_quarantine"]
            else:
                dec["decision"] = "unresolved"
                dec["decision_reason"] = "missing evidence or uncertain"

    # Build approved manifest rows (candidate)
    sha_to_rep_row: dict[str, dict[str, str]] = {}
    for sha, rows in hash_groups.items():
        sorted_rows = sorted(rows, key=lambda r: (r["work_id"], r["title"]))
        sha_to_rep_row[sha] = sorted_rows[0]

    candidate_manifest_rows: list[dict[str, Any]] = []
    for dec in decisions:
        if dec["decision"] != "approved":
            continue
        sha = dec["sha256"]
        rep = sha_to_rep_row[sha]
        composer = _text(rep.get("composer"))
        period = _composer_period(composer)
        work_id = _text(rep.get("work_id"))
        comp_id = _text(rep.get("composition_id"))
        title = _text(rep.get("title"))
        measures = _text(rep.get("measures")) or "1"
        try:
            bar_end = int(measures)
        except ValueError:
            bar_end = 1
        xml_path = _text(rep.get("xml_path"))
        rel = _relative_score_path(xml_path)
        suffix = Path(rel).suffix.casefold().lstrip(".")
        if suffix not in {"mxl", "musicxml", "xml", "mid", "midi"}:
            suffix = "musicxml"
        pid_raw = f"{dec['_analysis_id']}|{sha}|{work_id}".encode("utf-8")
        passage_id = f"exp_{hashlib.sha1(pid_raw).hexdigest()[:12]}"
        # Provenance: match score-version by sha + path, not work_id alone
        sv_match = find_score_version_for_selection(work_id, sha, rel, score_versions)
        if sv_match:
            sv_id = _text(sv_match.get("score_version_id"))
            source_dataset = _text(sv_match.get("source_dataset")) or _text(rep.get("origin")) or "canonical"
        else:
            # Sibling XML absent from canonical registrations: use derived identity
            # Preserve MIDI relationship: find at least one MIDI sv for work_id for context
            # Do not borrow unrelated canonical id
            sv_id = f"scv_derived_{sha[:12]}"
            source_dataset = _text(rep.get("origin")) or "ASAP sibling"
            # Document derived in decision? keep alias already
        row = {
            "passage_id": passage_id,
            "composer": composer,
            "period": period,
            "work_id": work_id,
            "movement_id": work_id,
            "recording_id": "",
            "score_path": rel,
            "mxl_path": rel if suffix in {"mxl", "musicxml", "xml"} else "",
            "audio_path": "",
            "midi_path": rel if suffix in {"mid", "midi"} else "",
            "alignment_path": "",
            "annotation_path": "",
            "bar_start": 1,
            "bar_end": bar_end,
            "score_format": suffix,
            "alignment_format": "",
            "rights_status": "licensed_for_research",
            "segment_type": "complete_movement",
            "annotation_type": "",
            "fully_aligned": "false",
            "eligible_for_clamp": "false",
            "availability_status": "needs_validation",
            "composition_id": comp_id,
            "work_title": title,
            "score_version_id": sv_id,
            "source_dataset": source_dataset,
            "source_record_ids": "",
            "analysis_work_id": dec["_analysis_id"],
        }
        candidate_manifest_rows.append(row)

    candidate_manifest_rows.sort(key=lambda r: (r["composer"], r["analysis_work_id"], r["passage_id"]))

    total_groups = len(hash_groups)
    approved = sum(1 for d in decisions if d["decision"] == "approved")
    quarantined = sum(1 for d in decisions if d["decision"] == "quarantined")
    already = sum(1 for d in decisions if d["decision"] == "already_represented")
    unresolved_uncertain = sum(1 for d in decisions if d["decision"] == "unresolved_uncertain_instrumentation")
    unresolved_missing = sum(1 for d in decisions if "missing" in d["decision"])
    unresolved_ambiguous = sum(1 for d in decisions if d["decision"] == "unresolved_ambiguous_edition")
    duplicate_excluded = sum(1 for d in decisions if d["decision"] == "excluded_not_selected")
    unresolved_other = sum(1 for d in decisions if d["decision"] == "unresolved" or d["decision"] == "unresolved_ambiguous_edition")

    # Input hashes for stale detection
    def _file_hash(p: Path) -> str:
        try:
            return _hash_file(p)
        except OSError:
            return ""

    input_hashes = {
        "usable_csv": _file_hash(args.usable_csv),
        "works_csv": _file_hash(args.works_csv),
        "compositions_csv": _file_hash(args.compositions_csv),
        "score_versions_csv": _file_hash(args.score_versions_csv),
        "current_main_manifest": _file_hash(args.current_main_manifest),
        "current_bach_manifest": _file_hash(args.current_bach_manifest),
    }
    # Also hash overrides if present
    if args.overrides_csv:
        input_hashes["overrides_csv"] = _file_hash(args.overrides_csv)
    if args.representative_overrides:
        input_hashes["representative_overrides"] = _file_hash(args.representative_overrides)

    # Summary
    validation_state = "metadata_only"
    if corpus_root is not None:
        validation_state = "needs_validation"  # will be updated after actual validation

    summary = {
        "total_hash_groups": total_groups,
        "total_usable_rows": len(usable),
        "approved_additions": approved,
        "quarantined_conflicts": quarantined,
        "already_represented": already,
        "unresolved_uncertain_instrumentation": unresolved_uncertain,
        "unresolved_missing_evidence": unresolved_missing,
        "unresolved_ambiguous_edition": unresolved_ambiguous,
        "excluded_not_selected": duplicate_excluded,
        "unresolved_other": unresolved_other,
        "output_manifest_rows": len(candidate_manifest_rows),
        "validation_state": validation_state,
        "validator_version": VALIDATOR_VERSION,
        "input_hashes": input_hashes,
        "note": "Candidates are grouped by identical XML checksums then by conservative candidate identity (composer|catalogue|movement). Approved additions are solo-piano, non-conflicting, not already represented, with explicit representative selection for ambiguous editions.",
    }

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    decision_fields = [
        "sha256",
        "composer",
        "record_count",
        "work_ids",
        "candidate_catalogue",
        "candidate_movement",
        "analysis_work_id",
        "decision",
        "decision_reason",
        "status",
        "uncertain_instrumentation",
        "quarantine_reason",
    ]
    decision_rows = []
    for d in sorted(decisions, key=lambda x: (x["composer"], x["sha256"])):
        row = {k: d.get(k, "") for k in decision_fields}
        row["sha256"] = d["sha256"]
        row["composer"] = d["composer"]
        row["record_count"] = d["record_count"]
        row["work_ids"] = d["work_ids"]
        row["candidate_catalogue"] = d["candidate_catalogue"]
        row["candidate_movement"] = d["candidate_movement"]
        row["analysis_work_id"] = d["analysis_work_id"]
        row["decision"] = d.get("decision", "")
        row["decision_reason"] = d.get("decision_reason", "")
        row["status"] = d["status"]
        row["uncertain_instrumentation"] = d["uncertain_instrumentation"]
        row["quarantine_reason"] = d["quarantine_reason"]
        decision_rows.append(row)

    _write_csv(args.output_dir / "decision.csv", decision_fields, decision_rows)

    alias_fields = ["analysis_work_id", "sha256", "work_id", "composition_id", "title", "xml_path", "candidate_catalogue", "candidate_movement"]
    alias_rows_sorted = sorted(alias_rows, key=lambda r: (r["analysis_work_id"] or "zzz", r["sha256"], r["work_id"]))
    _write_csv(args.output_dir / "alias_mapping.csv", alias_fields, alias_rows_sorted)

    with (args.output_dir / "conflict_report.json").open("w", encoding="utf-8") as handle:
        json.dump(conflicts, handle, indent=2, sort_keys=True, ensure_ascii=False)

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    manifest_fields = list(CORPUS_MANIFEST_FIELDS) + ["composition_id", "work_title", "score_version_id", "source_dataset", "source_record_ids", "analysis_work_id"]
    _write_csv(args.output_dir / "expanded_manifest.csv", manifest_fields, candidate_manifest_rows)

    # Also write expanded_manifest with empty if no approvals already handled (candidate is empty)

    # Validation if corpus_root provided
    validated_manifest_path = args.output_dir / "expanded_manifest.validated.csv"
    certificate_path = args.output_dir / "validation_certificate.json"

    if corpus_root is not None:
        # Validate every approved score
        per_score_errors: list[dict[str, Any]] = []
        validated_rows: list[dict[str, Any]] = []
        for dec in decisions:
            if dec["decision"] != "approved":
                continue
            sha = dec["sha256"]
            rep = sha_to_rep_row[sha]
            rel = _relative_score_path(_text(rep.get("xml_path")))
            bar_end = int(_text(rep.get("measures")) or "1")
            errors = _validate_score_file(corpus_root, rel, sha, bar_end)
            # Also check manifest consistency: score_version provenance already handled; verify suffix etc.
            # Check that score_version_id matches expected provenance (if canonical, should be find, if sibling derived)
            if errors:
                for msg in errors:
                    per_score_errors.append(
                        {
                            "sha256": sha,
                            "work_id": _text(rep.get("work_id")),
                            "score_path": rel,
                            "analysis_work_id": dec["_analysis_id"],
                            "error": msg,
                        }
                    )
            else:
                # Build validated row with ready status
                # Reuse candidate row but flip eligible/availability
                cand = next((r for r in candidate_manifest_rows if r["analysis_work_id"] == dec["_analysis_id"] and r["work_id"] == _text(rep.get("work_id"))), None)
                # Actually candidate rows are keyed by passage? Use constructed
                # Reconstruct validated row similarly but with ready status
                composer = _text(rep.get("composer"))
                period = _composer_period(composer)
                work_id = _text(rep.get("work_id"))
                comp_id = _text(rep.get("composition_id"))
                title = _text(rep.get("title"))
                # bar_end already
                suffix = Path(rel).suffix.casefold().lstrip(".")
                if suffix not in {"mxl", "musicxml", "xml"}:
                    suffix = "musicxml"
                pid_raw = f"{dec['_analysis_id']}|{sha}|{work_id}".encode("utf-8")
                passage_id = f"exp_{hashlib.sha1(pid_raw).hexdigest()[:12]}"
                sv_match = find_score_version_for_selection(work_id, sha, rel, score_versions)
                if sv_match:
                    sv_id = _text(sv_match.get("score_version_id"))
                    source_dataset = _text(sv_match.get("source_dataset")) or _text(rep.get("origin")) or "canonical"
                else:
                    sv_id = f"scv_derived_{sha[:12]}"
                    source_dataset = _text(rep.get("origin")) or "ASAP sibling"
                vrow = {
                    "passage_id": passage_id,
                    "composer": composer,
                    "period": period,
                    "work_id": work_id,
                    "movement_id": work_id,
                    "recording_id": "",
                    "score_path": rel,
                    "mxl_path": rel if suffix in {"mxl", "musicxml", "xml"} else "",
                    "audio_path": "",
                    "midi_path": rel if suffix in {"mid", "midi"} else "",
                    "alignment_path": "",
                    "annotation_path": "",
                    "bar_start": 1,
                    "bar_end": bar_end,
                    "score_format": suffix,
                    "alignment_format": "",
                    "rights_status": "licensed_for_research",
                    "segment_type": "complete_movement",
                    "annotation_type": "",
                    "fully_aligned": "false",
                    "eligible_for_clamp": "true",
                    "availability_status": "ready",
                    "composition_id": comp_id,
                    "work_title": title,
                    "score_version_id": sv_id,
                    "source_dataset": source_dataset,
                    "source_record_ids": "",
                    "analysis_work_id": dec["_analysis_id"],
                }
                validated_rows.append(vrow)

        if per_score_errors:
            # Update summary to reflect failure
            summary["validation_state"] = "validation_failed"
            summary["per_score_errors"] = per_score_errors
            summary["validated_manifest_rows"] = 0
            with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
            # Write a certification of failure? Don't overwrite validated outputs; keep previous intact
            # But write validation report for debugging
            with (args.output_dir / "validation_errors.json").open("w", encoding="utf-8") as handle:
                json.dump(per_score_errors, handle, indent=2, sort_keys=True)
            for err in per_score_errors:
                print(f"validation error for {err['score_path']} ({err['sha256'][:8]}): {err['error']}", file=sys.stderr)
            print(f"Validation failed for {len(per_score_errors)} score(s); validated manifest NOT updated.", file=sys.stderr)
            return 1

        # Success: publish validated manifest atomically and certificate
        validated_rows.sort(key=lambda r: (r["composer"], r["analysis_work_id"], r["passage_id"]))
        # Atomic write validated manifest
        _write_csv_atomic(validated_manifest_path, manifest_fields, validated_rows)

        # Compute hashes for certificate
        decision_hash = _hash_file(args.output_dir / "decision.csv")
        candidate_hash = _hash_file(args.output_dir / "expanded_manifest.csv")
        validated_hash = _hash_file(validated_manifest_path)
        # Score checksums as recorded at validation time
        score_checksums = {}
        for dec in decisions:
            if dec["decision"] != "approved":
                continue
            sha = dec["sha256"]
            rep = sha_to_rep_row[sha]
            rel = _relative_score_path(_text(rep.get("xml_path")))
            resolved = _resolve_score_path_safe(corpus_root, rel)
            score_checksums[sha] = _hash_file(resolved)

        certificate = {
            "validator_version": VALIDATOR_VERSION,
            "validation_state": "validated",
            "corpus_root": str(corpus_root),
            "input_hashes": input_hashes,
            "decision_hash": decision_hash,
            "candidate_manifest_hash": candidate_hash,
            "validated_manifest_hash": validated_hash,
            "validated_manifest_path": str(validated_manifest_path),
            "score_checksums": score_checksums,
            "approved_count": len(validated_rows),
            "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        }
        # Atomic write certificate
        tmp_cert = certificate_path.with_suffix(".tmp")
        with tmp_cert.open("w", encoding="utf-8") as handle:
            json.dump(certificate, handle, indent=2, sort_keys=True)
        tmp_cert.replace(certificate_path)

        summary["validation_state"] = "validated"
        summary["validated_manifest_rows"] = len(validated_rows)
        summary["validated_manifest_path"] = str(validated_manifest_path)
        summary["validation_certificate"] = str(certificate_path)
        with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)

    # Legacy --revalidate without corpus_root already errored earlier; if we are here with revalidate and corpus_root, validation already done

    print(
        f"Audit complete: {total_groups} hash groups, {approved} approved, {quarantined} quarantined, {already} already represented, {unresolved_uncertain+unresolved_missing+unresolved_ambiguous} unresolved"
    )
    print(f"Outputs written to {args.output_dir}")
    for k, v in summary.items():
        if k != "input_hashes":
            print(f"  {k}: {v}")
    if corpus_root is not None:
        if summary.get("validation_state") == "validated":
            print(f"Validated manifest: {validated_manifest_path}")
            print(f"Certificate: {certificate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
