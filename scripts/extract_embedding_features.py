#!/usr/bin/env python3
"""Extract resumable MusicXML features for every successfully embedded score passage.

Uses embedding tables as the selection source, taking rows with a valid
embedding_row. Preserves passage IDs, composer, work, source, and bar
coordinates. Reuses existing feature definitions and passage cropping without
modifying embeddings or source scores.

Features:
- Prefer existing prepared MusicXML windows from each bundle's .resume/prepared
  cache when provenance can be verified; otherwise reproduce the embedding
  pipeline's crop from the original score.
- Cache identity includes source content, window coordinates, and
  feature-extractor version (passage ID alone insufficient).
- Deduplicates identical extraction inputs while retaining a feature row per
  passage ID.
- Saves each completed extraction atomically; resume skips verified completed
  extractions and supports retrying failures.
- Records unsupported sources and failures explicitly; never silently treats
  MIDI-derived notation as original MusicXML.
- Validates selected bundles, required environment variables, and source
  availability before writing outputs; missing individual scores appear in
  coverage report.
Outputs: analysis-compatible feature CSV, failure CSV, coverage by composer
and window size. Changed sources or extractor versions invalidate stale cache.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from piano_clamp.embedding_io import EmbeddingIOError, read_embedding_bundle, sha256_path  # noqa: E402
from piano_clamp.features import FEATURE_FIELDS, FeatureError, MUSICXML_EXTENSIONS, extract_musicxml_features  # noqa: E402
from piano_clamp.prepare import PreparationError, crop_musicxml, mxl_musicxml  # noqa: E402

MUSICXML_SUPPORTED = MUSICXML_EXTENSIONS
MIDI_EXTENSIONS = {".mid", ".midi"}

FAILURE_FIELDS = (
    "passage_id",
    "bundle",
    "composer",
    "work_id",
    "composition_id",
    "movement_id",
    "recording_id",
    "bar_start",
    "bar_end",
    "window_bars",
    "passage_generation_method",
    "source_material",
    "source_path",
    "error_category",
    "error_message",
)

COVERAGE_FIELDS = (
    "composer",
    "window_bars",
    "total_passages",
    "feature_success",
    "failure_count",
    "unsupported_count",
    "missing_source_count",
    "success_rate",
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    t = Path(tmp)
    try:
        t.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(t, path)
    finally:
        t.unlink(missing_ok=True)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    t = Path(tmp)
    try:
        with t.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({f: "" if row.get(f) is None else str(row.get(f, "")) for f in fields})
        os.replace(t, path)
    finally:
        t.unlink(missing_ok=True)


def _extractor_version() -> str:
    # Feature-extractor version includes feature definitions + cropping logic.
    # Hash of src/piano_clamp/features.py and src/piano_clamp/prepare.py
    # so changed extractor or crop invalidates stale cache.
    h = hashlib.sha256()
    for rel in ("src/piano_clamp/features.py", "src/piano_clamp/prepare.py"):
        p = ROOT / rel
        if p.is_file():
            h.update(p.read_bytes())
        else:
            h.update(rel.encode())
    # Also include FEATURE_FIELDS to catch schema changes.
    h.update(",".join(FEATURE_FIELDS).encode())
    return h.hexdigest()


def _resolve_corpus_root(value: str | Path | None, environ: dict[str, str] | None = None) -> Path:
    env = environ if environ is not None else os.environ
    if value is not None and str(value).strip():
        text = str(value).strip()
        # Allow ${VAR} expansion
        if text.startswith("${") and text.endswith("}"):
            var = text[2:-1]
            resolved = env.get(var)
            if not resolved:
                raise ValueError(
                    f"corpus root references {var}, but that environment variable is not set. "
                    f"On the HPC: export PIANO_CLAMP_CORPUS_ROOT=/home/student/w/wshelor/share/datasets/classical-performance-corpus"
                )
            text = resolved
        else:
            # expand $VAR style
            text = os.path.expandvars(text)
            if "$" in text:
                raise ValueError(f"corpus root contains unresolved environment variable: {text}")
        candidate = Path(text).expanduser().resolve()
        if not candidate.is_dir() or not os.access(candidate, os.R_OK | os.X_OK):
            raise ValueError(f"corpus directory is missing or unreadable: {candidate}")
        return candidate
    # Try known env vars
    for var in ("PIANO_CLAMP_CORPUS_ROOT", "PIANO_CLAMP_DATASETS_ROOT", "PIANO_DATASETS_ROOT"):
        val = env.get(var)
        if val:
            candidate = Path(val).expanduser().resolve()
            if candidate.is_dir() and os.access(candidate, os.R_OK | os.X_OK):
                return candidate
            raise ValueError(f"corpus directory for {var} is missing or unreadable: {candidate}")
    # Try to infer from config? Fallback: error
    raise ValueError(
        "corpus root is required: pass --corpus-root or set PIANO_CLAMP_CORPUS_ROOT. "
        "On the HPC: export PIANO_CLAMP_CORPUS_ROOT=/home/student/w/wshelor/share/datasets/classical-performance-corpus"
    )


def _validate_bundle(path: Path) -> tuple[Any, list[dict[str, str]], dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"selected bundle does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"selected bundle is not a directory: {path}")
    try:
        matrix, rows, metadata = read_embedding_bundle(path)
    except EmbeddingIOError as exc:
        raise ValueError(f"bundle {path} is invalid: {exc}") from exc
    if "embedding_row" not in (rows[0].keys() if rows else []):
        # still check
        pass
    # Validate at least have table columns expected
    return matrix, rows, metadata


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None


def _resolve_source_file(corpus_root: Path, source_value: str) -> Path:
    p = Path(source_value)
    if p.is_absolute():
        return p.resolve()
    # Resolve relative to corpus_root, rejecting traversal
    candidate = (corpus_root / p).resolve()
    # Ensure within corpus_root
    if candidate != corpus_root and corpus_root not in candidate.parents:
        # Still allow if corpus_root is not the true root for this dataset (e.g., absolute path stored)
        # But we already handled absolute. For relative that escapes, it's an error.
        raise ValueError(f"source path escapes corpus_root: {source_value}")
    return candidate


def _classify_row(row: dict[str, str], corpus_root: Path) -> dict[str, Any]:
    passage_id = str(row.get("passage_id", "")).strip()
    composer = str(row.get("composer", "")).strip()
    # Preserve work identifiers
    work_id = str(row.get("work_id", row.get("work_title", ""))).strip()
    composition_id = str(row.get("composition_id", "")).strip()
    movement_id = str(row.get("movement_id", row.get("movement", ""))).strip()
    recording_id = str(row.get("recording_id", "")).strip()
    bar_start = _safe_int(row.get("bar_start", row.get("start_bar")))
    bar_end = _safe_int(row.get("bar_end", row.get("end_bar")))
    passage_method = str(row.get("passage_generation_method", "")).strip()
    source_material = str(row.get("source_material", row.get("embedding_modality", ""))).strip()
    # Explicitly treat performance_midi as unsupported (MIDI-derived)
    if source_material == "performance_midi":
        return {
            "category": "unsupported",
            "error_category": "unsupported_midi_source",
            "error_message": "MIDI-derived notation is not original MusicXML; source_material=performance_midi",
            "source_path_raw": str(row.get("source_path") or row.get("performance_midi_path") or ""),
        }
    # Collect candidate source path: prefer source_path, then score_path/mxl_path
    candidates = []
    for key in ("source_path", "score_path", "mxl_path", "relative_path"):
        v = str(row.get(key, "")).strip()
        if v:
            candidates.append(v)
    # If none, check midi paths to give unsupported rather than missing
    midi_candidate = str(row.get("midi_path", row.get("performance_midi_path", ""))).strip()
    source_raw = candidates[0] if candidates else midi_candidate
    if not source_raw:
        return {
            "category": "failure",
            "error_category": "missing_source_path",
            "error_message": "manifest records no symbolic source path",
            "source_path_raw": "",
        }
    suffix = Path(source_raw).suffix.lower()
    if suffix in MIDI_EXTENSIONS:
        return {
            "category": "unsupported",
            "error_category": "unsupported_midi_source",
            "error_message": f"MusicXML feature extraction does not support {suffix}: {source_raw} (MIDI-derived notation not treated as original MusicXML)",
            "source_path_raw": source_raw,
        }
    if suffix and suffix not in MUSICXML_SUPPORTED:
        return {
            "category": "unsupported",
            "error_category": "unsupported_source",
            "error_message": f"MusicXML feature extraction does not support {suffix}: {source_raw}",
            "source_path_raw": source_raw,
        }
    if not suffix:
        # No extension, treat as unsupported
        return {
            "category": "unsupported",
            "error_category": "unsupported_source",
            "error_message": f"source has no extension: {source_raw}",
            "source_path_raw": source_raw,
        }
    # Resolve file
    try:
        resolved = _resolve_source_file(corpus_root, source_raw)
    except ValueError as exc:
        return {
            "category": "failure",
            "error_category": "invalid_source_path",
            "error_message": str(exc),
            "source_path_raw": source_raw,
        }
    if not resolved.is_file():
        return {
            "category": "missing",
            "error_category": "missing_source",
            "error_message": f"source score does not exist: {resolved}",
            "source_path_raw": source_raw,
            "resolved": resolved,
        }
    if bar_start is None or bar_end is None:
        return {
            "category": "failure",
            "error_category": "invalid_bar_coordinates",
            "error_message": f"invalid bar coordinates bar_start={row.get('bar_start')} bar_end={row.get('bar_end')}",
            "source_path_raw": source_raw,
            "resolved": resolved,
        }
    if bar_start < 1 or bar_end < bar_start:
        return {
            "category": "failure",
            "error_category": "invalid_bar_coordinates",
            "error_message": f"invalid bar range {bar_start}-{bar_end}",
            "source_path_raw": source_raw,
            "resolved": resolved,
        }
    return {
        "category": "ok",
        "source_path_raw": source_raw,
        "resolved": resolved,
        "bar_start": bar_start,
        "bar_end": bar_end,
        "window_bars": bar_end - bar_start + 1,
        "passage_method": passage_method,
        "source_material": source_material,
        "work_id": work_id,
        "composition_id": composition_id,
        "movement_id": movement_id,
        "composer": composer,
        "recording_id": recording_id,
    }


def _crop_window_bytes(resolved: Path, bar_start: int, bar_end: int) -> bytes:
    suffix = resolved.suffix.lower()
    raw = resolved.read_bytes()
    if suffix == ".mxl":
        raw = mxl_musicxml(raw)
    window_bars = bar_end - bar_start + 1
    cropped, _, _ = crop_musicxml(raw, start_index=bar_start - 1, window_bars=window_bars)
    return cropped


def _find_prepared_candidate(bundle_paths: list[Path], passage_id: str) -> Path | None:
    for b in bundle_paths:
        prepared_dir = b / ".resume" / "prepared"
        if not prepared_dir.is_dir():
            continue
        # Prepared files are named <passage_id>.musicxml (or .xml)
        matches = list(prepared_dir.glob(f"{passage_id}.*"))
        if matches:
            # Prefer .musicxml
            for m in matches:
                if m.suffix.lower() in MUSICXML_SUPPORTED:
                    return m
            return matches[0]
    return None


def _extract_features_from_window_bytes(window_bytes: bytes, *, tmp_dir: Path | None = None) -> dict[str, Any]:
    # Write to temp file then extract (extract_musicxml_features expects Path).
    # Use a directory inside the output tree when available to avoid system /tmp for primary artifacts;
    # falls back to system temp only if no tmp_dir is provided (e.g., unit tests).
    parent = Path(tmp_dir) if tmp_dir is not None else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".musicxml", dir=str(parent) if parent is not None else None)
    os.close(fd)
    p = Path(tmp)
    try:
        p.write_bytes(window_bytes)
        return extract_musicxml_features(p)
    finally:
        p.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    # ----- Preflight: bundles -----
    bundle_paths = [Path(p).resolve() for p in args.bundle]
    if not bundle_paths:
        raise ValueError("at least one --bundle is required (explicitly selected bundles)")
    for bp in bundle_paths:
        _validate_bundle(bp)

    # ----- Preflight: corpus root / env -----
    corpus_root = _resolve_corpus_root(args.corpus_root)

    # Collect all successfully embedded passages
    selected: list[dict[str, Any]] = []
    for bp in bundle_paths:
        _, rows, metadata = read_embedding_bundle(bp)
        for row in rows:
            if str(row.get("embedding_row", "")).strip() != "":
                # Preserve row + bundle provenance
                selected.append({"bundle_path": bp, "bundle_name": bp.name, "row": dict(row), "metadata": metadata})

    if not selected:
        raise ValueError(f"no embedded passages (embedding_row) found in selected bundles: {bundle_paths}")

    # Sort deterministically for reproducibility
    selected.sort(key=lambda x: x["row"].get("passage_id", ""))

    if args.limit is not None:
        selected = selected[: args.limit]

    output_dir = Path(args.output_dir).resolve()
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"
    cache_dir = output_dir / ".cache"
    resume_dir = output_dir / ".resume"
    # Do not write outputs yet - preflight source availability is per-record, not fatal.
    # But ensure output parent exists after preflight env check.
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    resume_dir.mkdir(parents=True, exist_ok=True)

    if args.reset_cache:
        for child in cache_dir.glob("*"):
            if child.is_file():
                child.unlink()
        for child in resume_dir.glob("*"):
            if child.is_file():
                child.unlink()

    extractor_version = _extractor_version()

    # Classify rows
    immediate_failures: list[dict[str, Any]] = []
    ok_entries: list[dict[str, Any]] = []  # each entry: selected_index, row, bundle, classification
    for entry in selected:
        row = entry["row"]
        cls = _classify_row(row, corpus_root)
        entry["classification"] = cls
        if cls["category"] in ("unsupported", "missing", "failure"):
            # Record immediate failure (unsupported sources, missing files, invalid coords)
            passage_id = str(row.get("passage_id", ""))
            bar_start = _safe_int(row.get("bar_start", row.get("start_bar"))) or ""
            bar_end = _safe_int(row.get("bar_end", row.get("end_bar"))) or ""
            window_bars = ""
            if bar_start != "" and bar_end != "":
                try:
                    window_bars = int(bar_end) - int(bar_start) + 1
                except:  # noqa: E722
                    window_bars = ""
            immediate_failures.append({
                "passage_id": passage_id,
                "bundle": entry["bundle_name"],
                "composer": str(row.get("composer", "")),
                "work_id": str(row.get("work_id", row.get("work_title", ""))),
                "composition_id": str(row.get("composition_id", "")),
                "movement_id": str(row.get("movement_id", row.get("movement", ""))),
                "recording_id": str(row.get("recording_id", "")),
                "bar_start": bar_start,
                "bar_end": bar_end,
                "window_bars": window_bars,
                "passage_generation_method": str(row.get("passage_generation_method", "")),
                "source_material": str(row.get("source_material", row.get("embedding_modality", ""))),
                "source_path": cls.get("source_path_raw", ""),
                "error_category": cls.get("error_category", cls["category"]),
                "error_message": cls.get("error_message", ""),
            })
        else:
            ok_entries.append(entry)

    # Deduplicate identical extraction inputs
    # Cache key = hash(source_hash + bar_start + bar_end + extractor_version)
    digest_to_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    digest_info: dict[str, dict[str, Any]] = {}
    for entry in ok_entries:
        cls = entry["classification"]
        resolved: Path = cls["resolved"]
        bar_start = cls["bar_start"]
        bar_end = cls["bar_end"]
        source_hash = sha256_path(resolved)
        digest = hashlib.sha256(f"{source_hash}\x1f{bar_start}\x1f{bar_end}\x1f{extractor_version}".encode()).hexdigest()
        entry["digest"] = digest
        entry["source_hash"] = source_hash
        digest_to_entries[digest].append(entry)
        if digest not in digest_info:
            digest_info[digest] = {
                "resolved": resolved,
                "bar_start": bar_start,
                "bar_end": bar_end,
                "source_hash": source_hash,
                "digest": digest,
                "representative_passage_id": str(entry["row"].get("passage_id", "")),
            }

    # Process each unique digest: resumable, atomic per digest
    # Cache files: .cache/<digest>.json for success, .cache/<digest>.failure.json for failure
    success_features_by_digest: dict[str, dict[str, Any]] = {}
    failure_by_digest: dict[str, dict[str, Any]] = {}

    # Load existing cache state
    for digest in list(digest_info.keys()):
        success_path = cache_dir / f"{digest}.json"
        failure_path = cache_dir / f"{digest}.failure.json"
        if success_path.is_file() and not args.reset_cache:
            try:
                payload = json.loads(success_path.read_text(encoding="utf-8"))
                if payload.get("extractor_version") == extractor_version and payload.get("source_hash") == digest_info[digest]["source_hash"]:
                    # Also ensure source file still matches? Already checked via source_hash in digest
                    success_features_by_digest[digest] = payload["features"]
                    continue
                else:
                    # stale, treat as miss (will be overwritten)
                    pass
            except Exception:
                pass
        if failure_path.is_file() and not args.retry_failures and not args.reset_cache:
            try:
                payload = json.loads(failure_path.read_text(encoding="utf-8"))
                if payload.get("extractor_version") == extractor_version and payload.get("source_hash") == digest_info[digest]["source_hash"]:
                    failure_by_digest[digest] = payload
                    continue
            except Exception:
                pass

    # Determine which digests still need extraction
    pending_digests = [
        d for d in digest_info
        if d not in success_features_by_digest and d not in failure_by_digest
    ]
    # If retry_failures, we already excluded failure_by_digest hits, so pending includes previous failures
    # Sort for determinism
    pending_digests.sort()

    # Extraction loop with atomic per-completion saves
    for digest in pending_digests:
        info = digest_info[digest]
        resolved: Path = info["resolved"]
        bar_start: int = info["bar_start"]
        bar_end: int = info["bar_end"]
        representative = digest_to_entries[digest][0]
        passage_id = str(representative["row"].get("passage_id", ""))
        prepared_candidate = _find_prepared_candidate(bundle_paths, passage_id)
        # If multiple passage_ids share digest but have different prepared files,
        # search for any that matches; try each
        if not prepared_candidate:
            for ent in digest_to_entries[digest]:
                cand = _find_prepared_candidate(bundle_paths, str(ent["row"].get("passage_id", "")))
                if cand:
                    prepared_candidate = cand
                    break

        try:
            # Crop window bytes (authoritative)
            window_bytes = _crop_window_bytes(resolved, bar_start, bar_end)
            # Verify prepared provenance if candidate exists
            used_prepared = False
            if prepared_candidate and prepared_candidate.is_file():
                try:
                    prepared_bytes = prepared_candidate.read_bytes()
                    # Compare hash to window_bytes; if equal, provenance verified
                    if hashlib.sha256(prepared_bytes).hexdigest() == hashlib.sha256(window_bytes).hexdigest():
                        used_prepared = True
                        # Optionally use prepared_bytes (identical)
                        window_bytes = prepared_bytes
                except Exception:
                    used_prepared = False

            features = _extract_features_from_window_bytes(window_bytes, tmp_dir=cache_dir)
            # Validate feature fields
            for field in FEATURE_FIELDS:
                if field not in features:
                    raise FeatureError(f"missing feature field {field}")

            payload = {
                "digest": digest,
                "source_hash": info["source_hash"],
                "bar_start": bar_start,
                "bar_end": bar_end,
                "window_bars": bar_end - bar_start + 1,
                "extractor_version": extractor_version,
                "features": features,
                "used_prepared_cache": used_prepared,
                "prepared_candidate": str(prepared_candidate) if prepared_candidate else "",
                "source_path": str(resolved),
            }
            # Atomic write success cache
            success_path = cache_dir / f"{digest}.json"
            _atomic_json(success_path, payload)
            # Remove stale failure if any
            failure_path = cache_dir / f"{digest}.failure.json"
            if failure_path.is_file():
                failure_path.unlink(missing_ok=True)
            success_features_by_digest[digest] = features
        except (FeatureError, PreparationError, OSError, ValueError) as exc:
            # Record failure explicitly
            error_category = "unsupported_midi_source" if "MIDI" in str(exc) else "extraction_failed"
            # Distinguish unsupported vs general failure
            msg = str(exc)
            if "does not support" in msg and (".mid" in msg or "MIDI" in msg):
                error_category = "unsupported_midi_source"
            elif "unsupported" in msg.lower():
                error_category = "unsupported_source"
            payload = {
                "digest": digest,
                "source_hash": info["source_hash"],
                "bar_start": bar_start,
                "bar_end": bar_end,
                "window_bars": bar_end - bar_start + 1,
                "extractor_version": extractor_version,
                "error_category": error_category,
                "error_message": msg,
                "source_path": str(resolved),
                "prepared_candidate": str(prepared_candidate) if prepared_candidate else "",
            }
            failure_path = cache_dir / f"{digest}.failure.json"
            _atomic_json(failure_path, payload)
            # Remove stale success if any
            success_path = cache_dir / f"{digest}.json"
            if success_path.is_file():
                success_path.unlink(missing_ok=True)
            failure_by_digest[digest] = payload

    # Re-load any remaining failure cache that was hit but not in pending (for reporting)
    for digest in list(digest_info.keys()):
        if digest not in success_features_by_digest and digest not in failure_by_digest:
            failure_path = cache_dir / f"{digest}.failure.json"
            if failure_path.is_file():
                try:
                    payload = json.loads(failure_path.read_text(encoding="utf-8"))
                    failure_by_digest[digest] = payload
                except Exception:
                    pass

    # Build final feature rows (one per passage_id) and failure rows
    feature_rows: list[dict[str, Any]] = []
    digest_failure_rows: list[dict[str, Any]] = []

    for entry in ok_entries:
        digest = entry["digest"]
        row = entry["row"]
        bundle_name = entry["bundle_name"]
        if digest in success_features_by_digest:
            feats = success_features_by_digest[digest]
            record = {
                "passage_id": str(row.get("passage_id", "")),
                "bundle": bundle_name,
                "composer": str(row.get("composer", "")),
                "work_id": str(row.get("work_id", row.get("work_title", ""))),
                "composition_id": str(row.get("composition_id", "")),
                "movement_id": str(row.get("movement_id", row.get("movement", ""))),
                "recording_id": str(row.get("recording_id", "")),
                "bar_start": entry["classification"]["bar_start"],
                "bar_end": entry["classification"]["bar_end"],
                "window_bars": entry["classification"]["window_bars"],
                "passage_generation_method": str(row.get("passage_generation_method", "")),
                "source_material": str(row.get("source_material", row.get("embedding_modality", ""))),
                "source_path": entry["classification"]["source_path_raw"],
                "source_hash": entry["source_hash"],
                "digest": digest,
                "extractor_version": extractor_version,
            }
            record.update(feats)
            feature_rows.append(record)
        elif digest in failure_by_digest:
            payload = failure_by_digest[digest]
            digest_failure_rows.append({
                "passage_id": str(row.get("passage_id", "")),
                "bundle": bundle_name,
                "composer": str(row.get("composer", "")),
                "work_id": str(row.get("work_id", row.get("work_title", ""))),
                "composition_id": str(row.get("composition_id", "")),
                "movement_id": str(row.get("movement_id", row.get("movement", ""))),
                "recording_id": str(row.get("recording_id", "")),
                "bar_start": entry["classification"]["bar_start"],
                "bar_end": entry["classification"]["bar_end"],
                "window_bars": entry["classification"]["window_bars"],
                "passage_generation_method": str(row.get("passage_generation_method", "")),
                "source_material": str(row.get("source_material", row.get("embedding_modality", ""))),
                "source_path": entry["classification"]["source_path_raw"],
                "error_category": payload.get("error_category", "extraction_failed"),
                "error_message": payload.get("error_message", ""),
            })
        else:
            # Should not happen; treat as failed with generic
            digest_failure_rows.append({
                "passage_id": str(row.get("passage_id", "")),
                "bundle": bundle_name,
                "composer": str(row.get("composer", "")),
                "work_id": str(row.get("work_id", "")),
                "composition_id": str(row.get("composition_id", "")),
                "movement_id": str(row.get("movement_id", "")),
                "recording_id": str(row.get("recording_id", "")),
                "bar_start": entry["classification"]["bar_start"],
                "bar_end": entry["classification"]["bar_end"],
                "window_bars": entry["classification"]["window_bars"],
                "passage_generation_method": str(row.get("passage_generation_method", "")),
                "source_material": str(row.get("source_material", row.get("embedding_modality", ""))),
                "source_path": entry["classification"]["source_path_raw"],
                "error_category": "unknown",
                "error_message": "no cache entry found after extraction",
            })

    # Combine all failures
    all_failures = immediate_failures + digest_failure_rows
    # Sort for determinism
    feature_rows.sort(key=lambda r: r["passage_id"])
    all_failures.sort(key=lambda r: r["passage_id"])

    # Validate every selected passage has either features or failure
    total_selected = len(selected)
    accounted = len(feature_rows) + len(all_failures)
    if accounted != total_selected:
        raise RuntimeError(f"accounting mismatch: selected {total_selected} but features {len(feature_rows)} + failures {len(all_failures)} = {accounted}")

    # ----- Write outputs atomically -----
    # Feature CSV: analysis-compatible
    feature_fields = ["passage_id", "bundle", "composer", "work_id", "composition_id", "movement_id", "recording_id",
                      "bar_start", "bar_end", "window_bars", "passage_generation_method", "source_material", "source_path",
                      "source_hash", "digest", "extractor_version"] + list(FEATURE_FIELDS)
    feature_path = tables_dir / "musicxml_features.csv"
    _atomic_csv(feature_path, feature_rows, feature_fields)

    failure_path = tables_dir / "musicxml_feature_failures.csv"
    _atomic_csv(failure_path, all_failures, list(FAILURE_FIELDS))

    # Coverage by composer and window size
    # Group by (composer, window_bars)
    groups: dict[tuple[str, str], dict[str, int]] = {}
    # Also track per passage for grouping
    for entry in selected:
        row = entry["row"]
        composer = str(row.get("composer", "")).strip() or "unknown"
        bar_start = _safe_int(row.get("bar_start", row.get("start_bar")))
        bar_end = _safe_int(row.get("bar_end", row.get("end_bar")))
        wb = str(bar_end - bar_start + 1) if bar_start is not None and bar_end is not None else "unknown"
        key = (composer, wb)
        if key not in groups:
            groups[key] = {"total": 0, "success": 0, "failure": 0, "unsupported": 0, "missing": 0}
        groups[key]["total"] += 1

    for r in feature_rows:
        key = (str(r.get("composer", "")).strip() or "unknown", str(r.get("window_bars", "")))
        if key in groups:
            groups[key]["success"] += 1

    for r in all_failures:
        key = (str(r.get("composer", "")).strip() or "unknown", str(r.get("window_bars", "")))
        if key not in groups:
            groups[key] = {"total": 0, "success": 0, "failure": 0, "unsupported": 0, "missing": 0}
            # Should already exist but handle
            groups[key]["total"] = 1
        # Count category
        cat = str(r.get("error_category", ""))
        if cat.startswith("unsupported"):
            groups[key]["unsupported"] += 1
            groups[key]["failure"] += 1
        elif cat == "missing_source":
            groups[key]["missing"] += 1
            groups[key]["failure"] += 1
        else:
            groups[key]["failure"] += 1

    coverage_rows: list[dict[str, Any]] = []
    for (composer, window_bars), counts in sorted(groups.items()):
        total = counts["total"]
        success = counts["success"]
        failure = counts["failure"]
        coverage_rows.append({
            "composer": composer,
            "window_bars": window_bars,
            "total_passages": total,
            "feature_success": success,
            "failure_count": failure,
            "unsupported_count": counts["unsupported"],
            "missing_source_count": counts["missing"],
            "success_rate": round(success / total, 4) if total else 0.0,
        })
    coverage_path = tables_dir / "coverage_by_composer_and_window.csv"
    _atomic_csv(coverage_path, coverage_rows, list(COVERAGE_FIELDS))

    # Summary report
    summary = {
        "bundles": [str(p) for p in bundle_paths],
        "corpus_root": str(corpus_root),
        "output_dir": str(output_dir),
        "extractor_version": extractor_version,
        "total_selected_embedded_passages": total_selected,
        "feature_success": len(feature_rows),
        "failure_count": len(all_failures),
        "unsupported_count": sum(1 for r in all_failures if str(r.get("error_category", "")).startswith("unsupported")),
        "missing_source_count": sum(1 for r in all_failures if str(r.get("error_category", "")) == "missing_source"),
        "unique_extraction_inputs": len(digest_info),
        "deduplicated_inputs_saved": len(ok_entries) - len(digest_info) if ok_entries else 0,
        "pending_digests_this_run": len(pending_digests),
        "cache_hits": len(digest_info) - len(pending_digests),
        "immediate_failures": len(immediate_failures),
        "feature_csv": str(feature_path),
        "failure_csv": str(failure_path),
        "coverage_csv": str(coverage_path),
        "cache_dir": str(cache_dir),
    }
    summary_path = reports_dir / "extraction_summary.json"
    _atomic_json(summary_path, summary)

    # Validation: every selected passage has either features or failure (already)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", action="append", required=True, help="Embedding bundle directory (repeatable for explicitly selected bundles)")
    parser.add_argument("--corpus-root", help="Corpus root directory; defaults to $PIANO_CLAMP_CORPUS_ROOT or $PIANO_CLAMP_DATASETS_ROOT. Must be readable.")
    parser.add_argument("--output-dir", required=True, help="Output directory for feature tables and coverage reports")
    parser.add_argument("--limit", type=int, help="Limit number of passages (for small sample before full extraction)")
    parser.add_argument("--reset-cache", action="store_true", help="Clear existing feature cache before extraction (forces re-extraction)")
    parser.add_argument("--retry-failures", action="store_true", help="Retry previously failed extractions instead of skipping them")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output tables if they exist")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    # Guard against overwriting without --force if outputs exist and not resuming?
    # For resumable behavior, we allow overwrite of tables but cache is reused.
    # If tables exist and --force not given, we still update atomically (resume). So no guard.
    try:
        summary = run(args)
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
