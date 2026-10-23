"""Helpers for XML expansion audit: hash grouping, identity derivation, conflict detection.

Conservative candidate identities are derived from composer + catalogue + movement.
Recording labels (Live, Remastered, etc.) are ignored for grouping but title
similarity alone must not authorize a merge.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any

# Alternate Mozart Köchel mappings observed in the corpus.
# Multiple catalogue numbers for the same work are not automatically a conflict.
ALTERNATE_K_MAP: dict[str, str] = {
    "189h": "283",
    "189a": "283",
    "300i": "331",
    "300h": "331",
    "496a": "487",
    "496": "487",
    "385g": "397",
    "189h": "283",
}

# Recording-label patterns that may be ignored for candidate grouping.
_RECORDING_LABEL_PATTERNS = [
    r"\s*-\s*Live\b.*$",
    r"\s*\(Live[^)]*\)",
    r"\bLive\b",
    r"\s*-\s*Remastered\b.*$",
    r"\s*\(Remastered[^)]*\)",
    r"\bRemastered\b",
    r"\bDigitally\s+Remastered\b",
    r"\s*-\s*\d{4}\s*Recording\b",
    r"\s*\(\d{4}\s*Recording\)",
    r"\b\d{4}\s*Recording\b",
    r"\s*-\s*iTunes;\s*Bonus\b",
    r"\biTunes;\s*Bonus\b",
    r"\bBonus\b",
]

UNCERTAIN_INSTRUMENT_KEYWORDS = [
    "horn",
    "sehnsucht",
    "lied",
    "liriche",  # catch lieder
    "duo",
    "two pianos",
    "2 pianos",
    "orchestra",
    "concerto",
    "symphony",
    "violin",
    "cello",
    "string",
    "quartet",
    "choral",
    "voice",
]


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _ascii_fold(value: str) -> str:
    return unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().casefold()


def normalize_composer(value: object) -> str:
    key = _ascii_fold(_text(value))
    if "chopin" in key:
        return "Chopin"
    if "mozart" in key:
        return "Mozart"
    if "bach" in key:
        return "Bach"
    # Preserve original but strip
    return _text(value)


def strip_recording_labels(title: str) -> str:
    s = _text(title)
    # Remove parenthetical recording info first to avoid leftover dashes
    for pat in _RECORDING_LABEL_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*-\s*$", "", s).strip()
    s = re.sub(r"\s*\(\s*\)", "", s).strip()
    return s


def extract_k_catalogues(text: str) -> list[str]:
    """Extract raw K catalogue tokens from text, preserving colon-combined forms."""
    # Capture K or KV followed by numbers. Combined forms like K.487:496a (no space) or K.15a-15ss are kept.
    # Colon followed by movement (e.g., K.310: I.) must not be captured as part of catalogue.
    pattern = re.compile(r"\bK(?:V)?\.?\s*([0-9]+[A-Za-z0-9]*(?::[0-9]+[A-Za-z0-9]*)?(?:-[0-9]+[A-Za-z0-9]*)?)", re.IGNORECASE)
    results: list[str] = []
    for m in pattern.finditer(text):
        raw = m.group(1).strip()
        # Remove any trailing hyphen/colon artifacts
        raw = raw.strip(" :,-")
        if raw:
            results.append(raw)
    # Also capture bare numbers after "K.331 (300i)" where second number appears without K prefix but inside parens?
    # The above will capture KV? Already. For "(300i)" without K, we need secondary capture inside parentheses that looks like 300i
    # Search for parenthetical bare numbers that could be alternate: e.g., "(300i)" or "(189h)"
    # Only if text already had a K, treat bare numbers in parens as potential catalogue.
    paren_pat = re.compile(r"\(\s*([0-9]+[A-Za-z]*)\s*\)")
    for m in paren_pat.finditer(text):
        token = m.group(1).strip()
        # Heuristic: if token looks like K suffix (digits + letter), and we already have a K catalogue, consider it
        if re.fullmatch(r"[0-9]+[A-Za-z]*", token):
            # Avoid duplicates
            if token not in results:
                # Check if preceding text had K, then this is likely alternate
                if results:
                    results.append(token)
    return results


def normalize_k_token(token: str) -> str:
    """Normalize catalogue token and map known alternates to primary."""
    raw = token.strip().casefold().replace(" ", "")
    # Handle combined colon form "487:496a" -> keep as lower but map second part?
    # For combined, we treat whole as identity but also map alternate parts.
    # For now, if token contains colon, normalize each part via map and keep combined if both map to same primary?
    # Example: "487:496a" -> parts 487 and 496a ; 496a maps to 487, so combined maps to "487:487" -> simplify to "487"
    # But spec says Horn Duos K.487:496a is distinct combined, not to be simplified.
    # So keep combined as is lower.
    if ":" in raw:
        # Keep combined as is, but lower
        # Check if second part maps to first, then it's alternate pair, but we preserve original combined for identity
        return raw.lower()
    if "-" in raw:
        # Range like "15a-15ss" -> keep lower
        return raw.lower()
    # Simple token like "189h" -> map to primary
    primary = ALTERNATE_K_MAP.get(raw.lower(), raw.lower())
    return primary


def normalize_catalogue_for_comparison(tokens: list[str]) -> str | None:
    """Pick primary catalogue for comparison: last simple token normalized, or combined."""
    if not tokens:
        return None
    # Prefer last token that is not a range (contains hyphen with letters) if multiple?
    # For sketchbook, tokens = ["15a-15ss", "15b"] -> last "15b" is specific, pick it
    # For alternate, tokens = ["283", "189h"] -> last "189h" maps to "283", same as first
    # So pick last token normalized
    last = tokens[-1]
    return normalize_k_token(last)


def extract_movement_from_title(title: str, composer_key: str) -> str | None:
    """Extract normalized movement identifier from title (after stripping labels)."""
    s = strip_recording_labels(title)
    # Mozart / Bach: look for colon + roman or arabic movement number
    # e.g., ": III. Allegr" , ": 2. Andantino", ": 4. Polonaise"
    m = re.search(r":\s*([IVX]+|[0-9]+)\s*[\.\:]", s)
    if m:
        return normalize_movement(m.group(1))
    # Handle hyphen-separated sonata/movement like "11-3" (Piano Sonatas 11-3)
    m_h = re.search(r"\b(\d+)\s*-\s*(\d+)\b", s)
    if m_h:
        # Use second number as movement if context suggests sonata set
        if "sonata" in s.casefold() and composer_key in ("Mozart", "Chopin"):
            return normalize_movement(m_h.group(2))
    # For Chopin Sonata: "Sonata 2 1st", "Sonata 3 3rd"
    if composer_key == "Chopin":
        # Sonata case
        m2 = re.search(r"Sonata\s*(\d+)\s*(\d+)(?:st|nd|rd|th)?", s, re.IGNORECASE)
        if m2:
            # movement is second number (ordinal)
            return normalize_movement(m2.group(2))
        # Also handle "1st_no_repeat" in title? titles use "Sonata 2 1st" without _no_repeat
        # Check standalone ordinal
        m3 = re.search(r"\b(\d+)(?:st|nd|rd|th)\b", s, re.IGNORECASE)
        if m3 and "sonata" in s.casefold():
            return normalize_movement(m3.group(1))
        # Etudes: "Etudes op 10 5" -> op 10, piece 5 is movement
        m4 = re.search(r"Op\.?\s*(\d+)\s+(\d+)\b", s, re.IGNORECASE)
        if m4:
            # For Etudes, movement is second number
            # But need to ensure it's Etudes context
            if "etude" in s.casefold():
                return normalize_movement(m4.group(2))
        # Scherzos etc: "Scherzos 39" -> the number is catalogue, not movement; movement None for single-piece
        # For those, treat as no movement (single piece)
        # If contains "ballades", "scherzos" alone, movement is None (single)
        return None
    # For Mozart single piece without colon, try fallback: pattern "K. 4" with no movement -> None
    # Also check for "Variation" without movement number
    return None


def extract_movement_from_path(xml_path: str) -> str | None:
    """Extract movement identifier from xml_path leaf folder."""
    if not xml_path:
        return None
    p = Path(xml_path)
    # Leaf folder is parent of file, e.g., "3._Allegretto" or "1st_no_repeat"
    leaf = p.parent.name
    # Also consider the case where path contains hyphen movement like "11-3"
    # Check for hyphen movement pattern
    m_h = re.search(r"(\d+)\s*-\s*(\d+)", leaf)
    if m_h:
        return normalize_movement(m_h.group(2))
    # If leaf is like "score.mxl" parent is "3._Allegretto", good.
    # If path is like ".../Minuet_in_F_major,_K.4/score.mxl", leaf is "Minuet_in_F_major,_K.4" which contains composition, not movement
    # Heuristic: if leaf contains "K." then it's composition folder, not movement, unless it also starts with digit before dot.
    # Check if leaf starts with digit or roman + dot/underscore
    m = re.match(r"^\s*([0-9]+|[IVX]+)[\._]", leaf)
    if m:
        return normalize_movement(m.group(1))
    # Handle Chopin style "1st_no_repeat"
    m2 = re.match(r"^\s*([0-9]+)(?:st|nd|rd|th)", leaf, re.IGNORECASE)
    if m2:
        return normalize_movement(m2.group(1))
    # Also handle "1st_no_repeat" with full ordinal
    m3 = re.match(r"^\s*([0-9]+(?:st|nd|rd|th))", leaf, re.IGNORECASE)
    if m3:
        return normalize_movement(m3.group(1))
    return None


def extract_catalogue_from_path(xml_path: str) -> list[str]:
    """Extract K catalogues from path string."""
    return extract_k_catalogues(xml_path)


def normalize_movement(value: str) -> str | None:
    if not value:
        return None
    s = _text(value).strip().casefold().replace(".", "").strip()
    # Remove ordinal suffix
    s = re.sub(r"(st|nd|rd|th)$", "", s).strip()
    # Map roman to integer
    roman_map = {
        "i": "1",
        "ii": "2",
        "iii": "3",
        "iv": "4",
        "v": "5",
        "vi": "6",
        "vii": "7",
        "viii": "8",
        "ix": "9",
        "x": "10",
        "xi": "11",
        "xii": "12",
    }
    if s in roman_map:
        return roman_map[s]
    # Already numeric
    if re.fullmatch(r"[0-9]+", s):
        # Strip leading zeros
        return str(int(s))
    return s if s else None


def is_uncertain_instrumentation(title: str) -> bool:
    low = title.casefold()
    # Direct keywords trigger uncertain
    for kw in UNCERTAIN_INSTRUMENT_KEYWORDS:
        if kw in low:
            # For "duo" need to ensure not "duo" as part of other word? Already lower.
            # For horn duos, it's horn + duo, so uncertain.
            # For piano duo (two pianos) also uncertain, but we flag anyway.
            return True
    return False


def stable_analysis_work_id(composer: str, catalogue: str | None, movement: str | None) -> str:
    """Deterministic stable ID for candidate musical work."""
    comp = normalize_composer(composer)
    cat = (catalogue or "").strip().casefold()
    mov = (movement or "").strip().casefold()
    payload = "\x1f".join([comp.casefold(), cat, mov]).encode("utf-8")
    digest = hashlib.sha1(payload).hexdigest()[:12]
    # Prefix with composer initial for readability but keep stable
    prefix = comp[:2].casefold() if comp else "aw"
    # Use generic prefix "aw_" for analysis work
    return f"aw_{digest}"


def detect_title_path_catalogue_mismatch(title_cats: list[str], path_cats: list[str]) -> bool:
    """Return True if title and path catalogues conflict (not alternate)."""
    t_norm = normalize_catalogue_for_comparison(title_cats)
    p_norm = normalize_catalogue_for_comparison(path_cats)
    if not t_norm or not p_norm:
        return False  # missing evidence, not mismatch
    # Handle combined colon catalogues: compare as is
    # Also handle Horn Duos where path is "487:496a" and title "487:496a" same
    # But if title is "487" and path "487:496a", they differ but one contains colon?
    # Use normalized lower
    # If either contains colon, compare full normalized string
    if t_norm != p_norm:
        # Check alternate mapping: if they map to same primary via alternate map, not mismatch
        # Example t=283, p=189h -> both normalize to 283 -> equal, so no mismatch (already handled via normalize)
        # Since normalize already maps alternates, remaining inequality is genuine conflict
        return True
    return False


def detect_group_conflicts(rows: list[dict[str, Any]], works_lookup: dict[str, Any] = None, comps_lookup: dict[str, Any] = None) -> dict[str, Any]:
    """Detect conflicting catalogue or movement identities within a hash group and title-path mismatches.

    Returns dict with keys:
      conflicting_catalogues: bool
      conflicting_movements: bool
      title_path_catalogue_mismatches: list of row indices
      title_path_movement_mismatches: list of row indices
      catalogue_set: set
      movement_set: set
      missing_evidence: bool
    """
    cat_set: set[str] = set()
    mov_set: set[str] = set()
    title_path_cat_mism: list[int] = []
    title_path_mov_mism: list[int] = []
    missing = False

    # Track whether any title expects movement (contains ':')
    any_title_expects_movement = any(":" in _text(r.get("title")) for r in rows)
    any_sonata = any("sonata" in _text(r.get("title")).casefold() for r in rows)
    for idx, row in enumerate(rows):
        title = _text(row.get("title"))
        xml_path = _text(row.get("xml_path"))
        composer_key = normalize_composer(row.get("composer"))
        # Use derive to get normalized catalogue/movement for set construction
        der_cat, der_mov, _ = derive_candidate_identity(row, comps_lookup)
        if der_cat:
            cat_set.add(der_cat.casefold())
        if der_mov:
            mov_set.add(der_mov.casefold())

        # Title-path mismatch detection for K and BWV
        # K mismatch
        title_cats = extract_k_catalogues(title)
        path_cats = extract_catalogue_from_path(xml_path)
        # Fallback composition catalogue for K
        comp_id = _text(row.get("composition_id"))
        if not title_cats and comp_id and comps_lookup and comp_id in comps_lookup:
            comp_cat_raw = _text(comps_lookup[comp_id].get("catalogue"))
            if comp_cat_raw:
                title_cats = extract_k_catalogues(comp_cat_raw)
        t_norm = normalize_catalogue_for_comparison(title_cats)
        p_norm = normalize_catalogue_for_comparison(path_cats)
        if t_norm and p_norm and t_norm != p_norm:
            title_path_cat_mism.append(idx)
        # BWV mismatch for Bach
        if composer_key == "Bach":
            t_bwv = re.search(r"bwv\s*(\d+)", title.casefold())
            p_bwv = re.search(r"bwv[_ ]*(\d+)", xml_path.casefold())
            if t_bwv and p_bwv and t_bwv.group(1) != p_bwv.group(1):
                if idx not in title_path_cat_mism:
                    title_path_cat_mism.append(idx)

        mov_title = extract_movement_from_title(title, composer_key)
        mov_path = extract_movement_from_path(xml_path)
        if mov_title and mov_path and mov_title != mov_path:
            title_path_mov_mism.append(idx)

    # Missing evidence determination at group level
    if not cat_set:
        # No catalogue evidence across group
        # Check if any row expects catalogue (Mozart/Bach/Chopin)
        if any(normalize_composer(r.get("composer")) in ("Mozart", "Bach", "Chopin") for r in rows):
            missing = True
    if not mov_set:
        # No movement evidence but title expects movement (colon + movement pattern)
        expects_movement = any(re.search(r":\s*([IVX]+|[0-9]+)\s*[\.\:]", _text(r.get("title"))) for r in rows)
        if expects_movement and any(normalize_composer(r.get("composer")) == "Mozart" for r in rows):
            missing = True
        # Chopin sonata expects movement
        has_sonata = any("sonata" in _text(r.get("title")).casefold() for r in rows)
        if has_sonata and any(normalize_composer(r.get("composer")) == "Chopin" for r in rows):
            if not mov_set:
                missing = True

    conflicting_cats = len(cat_set) > 1
    conflicting_movs = len(mov_set) > 1

    # Also missing evidence if no catalogue at all but composer expects it
    # Already flagged

    return {
        "conflicting_catalogues": conflicting_cats,
        "conflicting_movements": conflicting_movs,
        "title_path_catalogue_mismatches": title_path_cat_mism,
        "title_path_movement_mismatches": title_path_mov_mism,
        "catalogue_set": sorted(cat_set),
        "movement_set": sorted(mov_set),
        "missing_evidence": missing,
    }


def derive_candidate_identity(row: dict[str, Any], comps_lookup: dict[str, Any] = None) -> tuple[str | None, str | None, str]:
    """Derive conservative candidate identity (catalogue, movement, evidence_summary).

    Returns (catalogue, movement, evidence_string) where catalogue/movement may be None if missing.
    Evidence_string describes sources used.
    """
    composer_key = normalize_composer(row.get("composer"))
    title = _text(row.get("title"))
    xml_path = _text(row.get("xml_path"))
    comp_id = _text(row.get("composition_id"))

    title_cats = extract_k_catalogues(title)
    path_cats = extract_catalogue_from_path(xml_path)
    if not title_cats and comp_id and comps_lookup and comp_id in comps_lookup:
        cc = _text(comps_lookup[comp_id].get("catalogue"))
        if cc:
            title_cats = extract_k_catalogues(cc)

    catalogue = normalize_catalogue_for_comparison(title_cats)
    # Fallback to path catalogue if title catalogue missing but path has one (specific piece)
    if not catalogue:
        catalogue = normalize_catalogue_for_comparison(path_cats)

    movement: str | None = None
    mov_title = extract_movement_from_title(title, composer_key)
    mov_path = extract_movement_from_path(xml_path)
    if mov_title:
        movement = mov_title
    elif mov_path:
        movement = mov_path
    else:
        movement = None

    # For Chopin, need catalogue handling beyond K
    if composer_key == "Chopin":
        s = strip_recording_labels(title).casefold()
        # Try to derive catalogue for Chopin
        # Sonata
        m = re.search(r"sonata\s*(\d+)", s)
        if m:
            catalogue = f"sonata{m.group(1)}"
            # movement already from ordinal
        elif "etudes" in s or "etude" in s:
            m2 = re.search(r"op\.?\s*(\d+)", s)
            if m2:
                catalogue = f"op{m2.group(1)}"
                # movement is etude number already
            else:
                catalogue = None
        elif "scherzo" in s:
            m2 = re.search(r"scherzo[s]?\s*(\d+)", s)
            if m2:
                catalogue = f"scherzo{m2.group(1)}"
        elif "ballade" in s:
            m2 = re.search(r"ballade[s]?\s*(\d+)", s)
            if m2:
                catalogue = f"ballade{m2.group(1)}"
        elif "prelude" in s:
            m2 = re.search(r"op\.?\s*(\d+)", s)
            if m2:
                catalogue = f"op{m2.group(1)}"
                # prelude number as movement?
                m3 = re.search(r"prelude[s]?\s*(?:op\.?\s*\d+\s*)?(\d+)", s)
                if m3:
                    movement = normalize_movement(m3.group(1))
        # Fallback: if catalogue still None, mark missing
        # For Sonata 2 etc., catalogue is sonata number, movement is ordinal
        # For Etudes, catalogue op, movement piece number
    elif composer_key == "Bach":
        s = strip_recording_labels(title).casefold()
        # Extract BWV catalogue
        m_bwv = re.search(r"bwv\s*(\d+)", s)
        if m_bwv:
            catalogue = f"bwv{m_bwv.group(1)}"
            # Distinguish prelude vs fugue as movement
            if "prelude" in s:
                movement = "prelude"
            elif "fugue" in s:
                movement = "fugue"
            else:
                # Keep existing movement if found via title/path extraction, else use generic
                if not movement:
                    # Try to infer from title: e.g., "Italian concerto" -> no bwv movement
                    movement = movement or None
        else:
            # No BWV found: could be Italian concerto or other without explicit BWV in title
            # Try to fallback to path BWV? Path may contain bwv folder
            path_bwv = re.search(r"bwv[_ ]*(\d+)", xml_path.casefold())
            if path_bwv:
                catalogue = f"bwv{path_bwv.group(1)}"
                if "prelude" in s:
                    movement = "prelude"
                elif "fugue" in s:
                    movement = "fugue"

    evidence = f"title_cats={title_cats} path_cats={path_cats} mov_title={mov_title} mov_path={mov_path}"
    return catalogue, movement, evidence


VALIDATOR_VERSION = "1.1.0"


def hash_file(path: Path) -> str:
    """SHA-256 of file contents, streamed."""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_corpus_path(corpus_root: Path, relative: str) -> Path:
    """Resolve a manifest-relative path safely within corpus_root."""
    root = Path(corpus_root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"corpus asset path must be repository-relative without traversal: {relative!r}")
    candidate = (root / rel).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"corpus asset path escapes corpus_root: {relative!r}")
    return candidate


def validate_representative_override(
    row: dict[str, Any],
    identity_to_shas: dict[str, list[str]],
    analysis_id_to_key: dict[str, str],
) -> None:
    """Validate a single representative-selection override. Raises ValueError."""
    aid = _text(row.get("analysis_work_id"))
    sha = _text(row.get("representative_sha256")).casefold()
    reason = _text(row.get("reason"))
    evidence = _text(row.get("evidence"))
    if not aid:
        raise ValueError("representative override missing analysis_work_id")
    if not sha:
        raise ValueError(f"representative override for {aid} missing representative_sha256")
    if not reason:
        raise ValueError(f"representative override for {aid} missing reason")
    if not evidence:
        raise ValueError(f"representative override for {aid} missing evidence")
    if aid not in analysis_id_to_key:
        raise ValueError(f"representative override for unknown analysis_work_id {aid!r}")
    shas = identity_to_shas.get(analysis_id_to_key[aid], [])
    normalized = {s.casefold() for s in shas}
    if sha not in normalized:
        raise ValueError(
            f"representative override for {aid} chooses sha {sha!r} which is not in group {sorted(shas)}"
        )


def find_score_version_for_selection(
    work_id: str,
    sha256: str,
    relative_path: str,
    score_versions: dict[str, dict[str, str]],
) -> dict[str, str] | None:
    """Find canonical score-version record matching work_id + sha + path, or None."""
    rel_norm = Path(relative_path).as_posix().lstrip("/")
    for sv in score_versions.values():
        if _text(sv.get("work_id")) != work_id:
            continue
        sv_sha = _text(sv.get("sha256")).casefold()
        if sv_sha and sv_sha != sha256.casefold():
            continue
        sv_path = _text(sv.get("source_file_path")).lstrip("/")
        # Compare path suffix-insensitively: exact relative match expected for XML registrations
        if sv_path and sv_path == rel_norm:
            # If sha present, already matched; if sha empty, still consider path match but require sha check earlier
            return sv
        # Also allow sha-only match when path not registered (e.g., MIDI) but we require both for XML
        if sv_sha == sha256.casefold() and sv_path == rel_norm:
            return sv
    # Fallback: sha match alone if path not available? Only if sv has same sha and work_id and is XML kind
    for sv in score_versions.values():
        if _text(sv.get("work_id")) == work_id and _text(sv.get("sha256")).casefold() == sha256.casefold():
            sv_path = _text(sv.get("source_file_path"))
            # If both are musicxml/mxl and path not checked, still treat as match only if format is musicxml-related
            # Require explicit path equality for provenance safety; do not fallback to sha-only for XML
            # So no fallback - return None to trigger derived handling
            pass
    return None


def validate_override(override: dict[str, Any], group_rows: list[dict[str, Any]], works_lookup: dict, comps_lookup: dict) -> None:
    """Validate override for a conflicting hash group. Raises ValueError if invalid or contradictory."""
    sha = _text(override.get("sha256"))
    reason = _text(override.get("reason"))
    evidence = _text(override.get("evidence"))
    resolved_catalogue = _text(override.get("resolved_catalogue"))
    resolved_movement = _text(override.get("resolved_movement"))
    if not sha or not reason:
        raise ValueError(f"override for {sha} must have sha256 and reason")
    if not evidence:
        raise ValueError(f"override for {sha} missing evidence")
    if not group_rows:
        raise ValueError(f"override sha {sha} not found in hash groups")
    # Check that resolved values are consistent with retained evidence if provided
    # Retained evidence: we require that resolved catalogue/movement must be one of the catalogue_set/movement_set or explicitly justified
    conflicts = detect_group_conflicts(group_rows, works_lookup, comps_lookup)
    cat_set = set(conflicts["catalogue_set"])
    mov_set = set(conflicts["movement_set"])
    if resolved_catalogue:
        rc = normalize_k_token(resolved_catalogue) if any("k" in resolved_catalogue.casefold() for _ in [1]) else resolved_catalogue.casefold()
        # For Mozart, resolved catalogue should be normalized K token
        # If cat_set has conflicting catalogues, resolved must be one of them; if not, it's contradictory
        if cat_set and rc not in {c.casefold() for c in cat_set}:
            # Allow alternate mapping resolution? Check normalized
            normalized_rc = normalize_k_token(resolved_catalogue)
            if normalized_rc not in {normalize_k_token(c) for c in cat_set}:
                raise ValueError(f"override catalogue {resolved_catalogue} contradicts evidence {cat_set} for sha {sha}")
    if resolved_movement:
        rm = normalize_movement(resolved_movement) or resolved_movement.casefold()
        if mov_set and rm not in {m.casefold() for m in mov_set}:
            raise ValueError(f"override movement {resolved_movement} contradicts evidence {mov_set} for sha {sha}")
    # Also check title-path mismatches: if override does not address mismatches, still considered contradictory?
    # For now, require that if there are title-path mismatches, override must explicitly acknowledge
    if conflicts["title_path_catalogue_mismatches"] or conflicts["title_path_movement_mismatches"]:
        # Override must provide reason; we already check reason, but also ensure resolved values address path
        pass
