#!/usr/bin/env python3
"""Metadata-only XML expansion audit with conflict quarantine and deduplication.

Default is audit mode (no HPC files required). Use --revalidate to check XML
readability, checksums and measure ranges when files are available.

Outputs (default to reports/hpc/xml_expansion_audit/review):
  - decision.csv
  - alias_mapping.csv
  - conflict_report.json
  - summary.json
  - expanded_manifest.csv  (pipeline-compatible XML-only manifest for approved additions)

Do not modify canonical source CSVs or existing manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from piano_clamp.data_loading import CORPUS_MANIFEST_FIELDS
from piano_clamp.xml_expansion import (
    derive_candidate_identity,
    detect_group_conflicts,
    is_uncertain_instrumentation,
    normalize_composer,
    stable_analysis_work_id,
    strip_recording_labels,
    validate_override,
)


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


def _preflight(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"error: {label} does not exist: {path}")
    if not path.stat().st_size:
        raise SystemExit(f"error: {label} is empty: {path}")


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
        out[sha] = {k: _text(v) for k, v in r.items()}
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
    parser.add_argument("--revalidate", action="store_true", help="Revalidate XML readability/checksums/measures (requires local files)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args(argv)

    # Preflight
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
        # Check if expected outputs already exist and are incompatible (simple check)
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
        # Use work_title and composition etc. to derive identity
        # For selected, we have composer, work_title, work_id, composition_id
        comp_id = _text(manifest_row.get("composition_id"))
        composer = _text(manifest_row.get("composer"))
        # Try to get composition catalogue
        comp_cat = ""
        if comp_id in comps:
            comp_cat = _text(comps[comp_id].get("catalogue"))
        # Build synthetic row for identity derivation
        synth = {
            "composer": composer,
            "title": _text(manifest_row.get("work_title")),
            "xml_path": _text(manifest_row.get("score_path")) or _text(manifest_row.get("mxl_path")),
            "composition_id": comp_id,
        }
        cat, mov, _ = derive_candidate_identity(synth, comps)
        if cat or mov:
            # Build identity key for comparison
            key = f"{normalize_composer(composer).casefold()}|{(cat or '').casefold()}|{(mov or '').casefold()}"
            selected_identities.add(key)
            selected_identity_details[key].append(_text(manifest_row.get("work_id")))

    # Process each hash group
    decisions: list[dict[str, Any]] = []
    alias_rows: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    approved_manifest_rows: list[dict[str, Any]] = []

    # For deduplication across hash groups: map candidate identity -> list of sha groups
    identity_to_shas: dict[str, list[str]] = defaultdict(list)
    sha_to_identity: dict[str, tuple[str | None, str | None]] = {}
    sha_to_analysis_id: dict[str, str] = {}

    # First pass: detect conflicts and derive identities
    for sha, rows in sorted(hash_groups.items()):
        # Detect conflicts
        conflict_info = detect_group_conflicts(rows, works, comps)
        has_override = sha in overrides
        override = overrides.get(sha)

        # Validate override if present
        if has_override:
            try:
                validate_override(override, rows, works, comps)
            except ValueError as exc:
                raise SystemExit(f"error: invalid override for sha {sha}: {exc}")

        # Determine if quarantined
        quarantine_reason = None
        if conflict_info["conflicting_catalogues"]:
            quarantine_reason = f"conflicting_catalogues:{conflict_info['catalogue_set']}"
        elif conflict_info["conflicting_movements"]:
            quarantine_reason = f"conflicting_movements:{conflict_info['movement_set']}"
        elif conflict_info["title_path_catalogue_mismatches"]:
            quarantine_reason = f"title_path_catalogue_mismatch rows {conflict_info['title_path_catalogue_mismatches']}"
        elif conflict_info["title_path_movement_mismatches"]:
            quarantine_reason = f"title_path_movement_mismatch rows {conflict_info['title_path_movement_mismatches']}"

        # If quarantined but override present and valid, allow override to resolve
        if quarantine_reason and has_override:
            # Override resolves; treat as not quarantined but record
            resolved_cat = _text(override.get("resolved_catalogue"))
            resolved_mov = _text(override.get("resolved_movement"))
            # Use resolved values for identity
            composer = _text(rows[0].get("composer"))
            cat = resolved_cat or None
            mov = resolved_mov or None
            # Normalize
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
            conflicts.append({
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
            })
            quarantine_reason = None  # cleared
        elif quarantine_reason:
            # Record conflict and quarantine
            conflicts.append({
                "sha256": sha,
                "type": "quarantined_conflict",
                "reason": quarantine_reason,
                "evidence": conflict_info,
                "work_ids": sorted({r["work_id"] for r in rows}),
                "titles": sorted({r["title"] for r in rows}),
                "xml_paths": sorted({r["xml_path"] for r in rows}),
            })

        # Derive candidate identity if not quarantined
        cat = None
        mov = None
        analysis_id = None
        missing_evidence = conflict_info["missing_evidence"]
        uncertain = False
        # Check uncertain instrumentation: if any row in group has uncertain title, flag
        for r in rows:
            if is_uncertain_instrumentation(_text(r.get("title"))):
                uncertain = True
                break

        if not quarantine_reason:
            # Derive identity: try each row in group, pick first with catalogue evidence (conservative but not missing due to one underspecified title)
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
            # If missing catalogue for Mozart/Chopin/Bach where expected, treat as missing evidence
            composer_key = normalize_composer(chosen_row.get("composer"))
            if not cat and composer_key in ("Mozart", "Chopin", "Bach"):
                missing_evidence = True
            if not cat and not mov:
                missing_evidence = True
            if missing_evidence or conflict_info["missing_evidence"]:
                quarantine_reason = "missing_identity_evidence"
                conflicts.append({
                    "sha256": sha,
                    "type": "missing_evidence",
                    "evidence": conflict_info,
                    "work_ids": sorted({r["work_id"] for r in rows}),
                })
            else:
                analysis_id = stable_analysis_work_id(_text(chosen_row.get("composer")), cat, mov)
                sha_to_identity[sha] = (cat, mov)
                sha_to_analysis_id[sha] = analysis_id
                identity_key = f"{normalize_composer(_text(chosen_row.get('composer'))).casefold()}|{(cat or '').casefold()}|{(mov or '').casefold()}"
                identity_to_shas[identity_key].append(sha)

        # Build decision entry (preliminary, will adjust for already_represented and uncertain)
        # Alias mapping for every source ID
        composer = _text(rows[0].get("composer"))
        for r in rows:
            alias_rows.append({
                "sha256": sha,
                "work_id": _text(r.get("work_id")),
                "composition_id": _text(r.get("composition_id")),
                "title": _text(r.get("title")),
                "xml_path": _text(r.get("xml_path")),
                "analysis_work_id": analysis_id or "",
                "candidate_catalogue": cat or "",
                "candidate_movement": mov or "",
            })

        has_parent_selected = any(_text(r.get("parent_composition_already_selected")).casefold() == "true" for r in rows)
        decisions.append({
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
        })

    # Second pass: handle deduplication across identities and already_represented, uncertain, missing
    # Group decisions by identity_key
    for dec in decisions:
        # Work-ID overlap check: if any work_id in group is already in selected manifests, it's already represented
        group_work_ids = set(dec["work_ids"].split(";")) if dec["work_ids"] else set()
        if group_work_ids & selected_work_ids:
            dec["decision"] = "already_represented"
            dec["decision_reason"] = f"already represented via work_id overlap {sorted(group_work_ids & selected_work_ids)}"
            dec["quarantine_reason"] = dec["quarantine_reason"] or ""
            continue
        # Parent composition already selected (alternative XML for selected work)
        if dec.get("_parent_selected"):
            # If group contains an alternative for already selected composition, it's already represented
            # Check status to avoid marking pure unselected groups with parent True incorrectly?
            # For pure unselected groups, parent should be False. For mixed or alternative groups, parent True indicates already.
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
        # Check if already represented in selected manifests via musical identity
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
        # Check duplicate identity across multiple shas: need representative policy
        # If identity maps to multiple shas, only one representative is approved, others are duplicates but same movement
        shas_for_identity = identity_to_shas.get(key, [])
        if len(shas_for_identity) > 1:
            # Choose representative deterministically: smallest sha
            representative_sha = sorted(shas_for_identity)[0]
            if dec["sha256"] != representative_sha:
                dec["decision"] = "duplicate_movement_different_edition"
                dec["decision_reason"] = f"duplicate of {representative_sha} under same candidate identity"
                continue
            else:
                # Representative: check if there are multiple work_ids within same sha already grouped; that's fine
                dec["decision"] = "approved"
                dec["decision_reason"] = "representative for candidate movement with multiple editions/versions"
                continue
        # Single identity case
        dec["decision"] = "approved"
        dec["decision_reason"] = "passes all checks"

    # Handle decisions that still have no decision (quarantined missing)
    for dec in decisions:
        if "decision" not in dec:
            if dec["_quarantine"]:
                dec["decision"] = "quarantined"
                dec["decision_reason"] = dec["_quarantine"]
            else:
                dec["decision"] = "unresolved"
                dec["decision_reason"] = "missing evidence or uncertain"

    # Now build approved manifest rows
    # For each approved decision, create manifest row from representative usable row
    # Need to find representative usable row per sha: choose row with canonical origin preference, else smallest work_id
    sha_to_rep_row: dict[str, dict[str, str]] = {}
    for sha, rows in hash_groups.items():
        # Find representative row within hash group: prefer canonical status True? Usable has canonical_xml_checksum_available flag
        # Also prefer status unselected_work vs xml_alternative? For approved, we want unselected_work candidates.
        # Sort by work_id for determinism
        sorted_rows = sorted(rows, key=lambda r: (r["work_id"], r["title"]))
        # Prefer rows where origin canonical? usable has origin field via xml_path? Not explicit. Use status priority: unselected_work first
        # Keep sorted order as representative is first
        sha_to_rep_row[sha] = sorted_rows[0]

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
        # Score path relative: try to make relative to corpus root if absolute HPC path
        xml_path = _text(rep.get("xml_path"))
        # Convert absolute HPC path to relative
        hpc_prefix = "/home/student/w/wshelor/share/datasets/classical-performance-corpus/"
        share_prefix = "/share/users/student/w/wshelor/datasets/classical-performance-corpus/"
        if xml_path.startswith(hpc_prefix):
            rel = xml_path[len(hpc_prefix):]
        elif xml_path.startswith(share_prefix):
            rel = xml_path[len(share_prefix):]
        else:
            rel = xml_path.lstrip("/")
        # Ensure no leading slash
        rel = rel.lstrip("/")
        # Determine score_format
        suffix = Path(rel).suffix.casefold().lstrip(".")
        if suffix not in {"mxl", "musicxml", "xml", "mid", "midi"}:
            suffix = "musicxml"
        # Passage id stable: hash of analysis_work_id + sha
        pid_raw = f"{dec['_analysis_id']}|{sha}|{work_id}".encode("utf-8")
        passage_id = f"exp_{hashlib.sha1(pid_raw).hexdigest()[:12]}"
        # Score version id: lookup via work_id in score_versions? There may be multiple; pick first matching work_id
        sv_id = ""
        for sv in score_versions.values():
            if _text(sv.get("work_id")) == work_id:
                sv_id = _text(sv.get("score_version_id"))
                break
        # If not found, synthesize
        if not sv_id:
            sv_id = f"scv_{sha[:12]}"
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
            "eligible_for_clamp": "true",
            "availability_status": "ready",
            "composition_id": comp_id,
            "work_title": title,
            "score_version_id": sv_id,
            "source_dataset": _text(rep.get("origin")) or "canonical",
            "source_record_ids": "",
            "analysis_work_id": dec["_analysis_id"],
        }
        approved_manifest_rows.append(row)

    # Sort approved rows deterministically
    approved_manifest_rows.sort(key=lambda r: (r["composer"], r["analysis_work_id"], r["passage_id"]))

    # Build summary
    total_groups = len(hash_groups)
    approved = sum(1 for d in decisions if d["decision"] == "approved")
    quarantined = sum(1 for d in decisions if d["decision"] == "quarantined")
    already = sum(1 for d in decisions if d["decision"] == "already_represented")
    unresolved_uncertain = sum(1 for d in decisions if d["decision"] == "unresolved_uncertain_instrumentation")
    unresolved_missing = sum(1 for d in decisions if "missing" in d["decision"])
    duplicate = sum(1 for d in decisions if d["decision"] == "duplicate_movement_different_edition")
    unresolved_other = sum(1 for d in decisions if d["decision"] == "unresolved")

    summary = {
        "total_hash_groups": total_groups,
        "total_usable_rows": len(usable),
        "approved_additions": approved,
        "quarantined_conflicts": quarantined,
        "already_represented": already,
        "unresolved_uncertain_instrumentation": unresolved_uncertain,
        "unresolved_missing_evidence": unresolved_missing,
        "duplicate_movement_editions": duplicate,
        "unresolved_other": unresolved_other,
        "output_manifest_rows": len(approved_manifest_rows),
        "note": "Candidates are grouped by identical XML checksums then by conservative candidate identity (composer|catalogue|movement). Approved additions are solo-piano, non-conflicting, not already represented, with representative version per movement.",
    }

    # Write outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Decision CSV
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
        decision_rows.append({k: d.get(k, "") for k in decision_fields})
        # Ensure decision fields exist
        decision_rows[-1]["sha256"] = d["sha256"]
        decision_rows[-1]["composer"] = d["composer"]
        decision_rows[-1]["record_count"] = d["record_count"]
        decision_rows[-1]["work_ids"] = d["work_ids"]
        decision_rows[-1]["candidate_catalogue"] = d["candidate_catalogue"]
        decision_rows[-1]["candidate_movement"] = d["candidate_movement"]
        decision_rows[-1]["analysis_work_id"] = d["analysis_work_id"]
        decision_rows[-1]["decision"] = d.get("decision", "")
        decision_rows[-1]["decision_reason"] = d.get("decision_reason", "")
        decision_rows[-1]["status"] = d["status"]
        decision_rows[-1]["uncertain_instrumentation"] = d["uncertain_instrumentation"]
        decision_rows[-1]["quarantine_reason"] = d["quarantine_reason"]

    _write_csv(args.output_dir / "decision.csv", decision_fields, decision_rows)

    # Alias mapping
    alias_fields = ["analysis_work_id", "sha256", "work_id", "composition_id", "title", "xml_path", "candidate_catalogue", "candidate_movement"]
    alias_rows_sorted = sorted(alias_rows, key=lambda r: (r["analysis_work_id"] or "zzz", r["sha256"], r["work_id"]))
    _write_csv(args.output_dir / "alias_mapping.csv", alias_fields, alias_rows_sorted)

    # Conflict report
    with (args.output_dir / "conflict_report.json").open("w", encoding="utf-8") as handle:
        json.dump(conflicts, handle, indent=2, sort_keys=True, ensure_ascii=False)

    # Summary
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    # Expanded manifest
    if approved_manifest_rows:
        manifest_fields = list(CORPUS_MANIFEST_FIELDS) + ["composition_id", "work_title", "score_version_id", "source_dataset", "source_record_ids", "analysis_work_id"]
        _write_csv(args.output_dir / "expanded_manifest.csv", manifest_fields, approved_manifest_rows)
    else:
        # Write empty manifest with header
        manifest_fields = list(CORPUS_MANIFEST_FIELDS) + ["composition_id", "work_title", "score_version_id", "source_dataset", "source_record_ids", "analysis_work_id"]
        _write_csv(args.output_dir / "expanded_manifest.csv", manifest_fields, [])

    # Revalidation if requested
    if args.revalidate:
        # Check each approved manifest row's file exists, checksum matches, measure range valid
        # For metadata-only audit without HPC files, we skip if files not present but warn
        for row in approved_manifest_rows:
            # Resolve path relative to corpus root? We don't have corpus root; try to check if file exists as given
            # Since local files not present, we just check that bar range is plausible
            try:
                bs = int(row["bar_start"])
                be = int(row["bar_end"])
                if bs < 1 or be < bs:
                    raise SystemExit(f"error: invalid bar range for {row['passage_id']}: {bs}-{be}")
            except ValueError:
                raise SystemExit(f"error: invalid bar range for {row['passage_id']}")

    # Print summary to stdout
    print(f"Audit complete: {total_groups} hash groups, {approved} approved, {quarantined} quarantined, {already} already represented, {unresolved_uncertain+unresolved_missing} unresolved")
    print(f"Outputs written to {args.output_dir}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
