"""Portable configuration and read-only loading for the external corpus."""

from __future__ import annotations

import csv
import os
import re
import unicodedata
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


CORPUS_MANIFEST_FIELDS = (
    "passage_id",
    "composer",
    "period",
    "work_id",
    "movement_id",
    "recording_id",
    "score_path",
    "mxl_path",
    "audio_path",
    "midi_path",
    "alignment_path",
    "annotation_path",
    "bar_start",
    "bar_end",
    "score_format",
    "alignment_format",
    "rights_status",
    "segment_type",
    "annotation_type",
    "fully_aligned",
    "eligible_for_clamp",
    "availability_status",
)

PROMPT_FIELDS = (
    "prompt_id",
    "family",
    "subfamily",
    "prompt_text",
    "polarity",
    "composer_reference",
    "interpretive_level",
    "enabled",
    "notes",
)

SYMBOLIC_EXTENSIONS = {".mxl", ".musicxml", ".xml", ".mid", ".midi"}
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}
_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
RECORDING_REGISTRATION_FIELDS = (
    "composer",
    "period",
    "composition_id",
    "work_id",
    "movement_id",
    "recording_id",
    "audio_path",
    "rights_status",
    "audio_origin",
)
SYMBOLIC_REGISTRATION_FIELDS = (
    "composer",
    "period",
    "composition_id",
    "work_id",
    "movement_id",
    "score_path",
    "mxl_path",
    "midi_path",
    "score_format",
    "score_version_id",
)


class PipelineConfigurationError(ValueError):
    """Raised when a pipeline YAML cannot be resolved portably."""


class CorpusManifestError(ValueError):
    """Raised when the external export violates its declared schema."""


class PromptBankError(ValueError):
    """Raised when the versioned prompt bank is malformed."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml_mapping(path: Path, *, field: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise PipelineConfigurationError(f"{field} must be a YAML mapping")
    return value


def _merge_overlay(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_overlay(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _expand_path(value: Any, *, field: str, environ: Mapping[str, str]) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PipelineConfigurationError(f"{field} must be a non-empty path")
    text = value.strip()
    match = _ENV_REFERENCE.fullmatch(text)
    if match:
        variable = match.group(1)
        resolved = environ.get(variable)
        if not resolved:
            raise PipelineConfigurationError(
                f"{field} references {variable}, but that environment variable is not set"
            )
        text = resolved
    else:
        text = os.path.expandvars(text)
        if "$" in text:
            raise PipelineConfigurationError(f"{field} contains an unresolved environment variable")
    return Path(text).expanduser()


def _repo_path(
    value: Any,
    *,
    field: str,
    root: Path,
    environ: Mapping[str, str],
) -> Path:
    path = _expand_path(value, field=field, environ=environ)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _corpus_path(root: Path, value: Any, *, field: str, environ: Mapping[str, str]) -> Path:
    path = _expand_path(value, field=field, environ=environ)
    candidate = path.resolve() if path.is_absolute() else (root / path).resolve()
    if candidate != root and root not in candidate.parents:
        raise PipelineConfigurationError(f"{field} escapes corpus_root")
    return candidate


def load_pipeline_config(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    root: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Load the embedding-pipeline YAML and resolve all paths.

    Corpus paths remain below the configured read-only corpus root. Repository
    assets and generated-output roots are resolved relative to this repository,
    never relative to the caller's current directory.
    """

    env = os.environ if environ is None else environ
    repo = Path(root).resolve() if root else repository_root()
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise PipelineConfigurationError(f"configuration file does not exist: {config_path}")
    config = _load_yaml_mapping(config_path, field="configuration")
    overlay_path = config_path.with_name("paths.yaml")
    if (
        config_path.name != "paths.yaml"
        and config.get("use_local_paths_overlay", True)
        and overlay_path.is_file()
    ):
        config = _merge_overlay(config, _load_yaml_mapping(overlay_path, field="paths.yaml"))

    corpus_root = _repo_path(
        config.get("corpus_root"), field="corpus_root", root=repo, environ=env
    )
    config["corpus_root"] = str(corpus_root)
    for name in ("score_root", "audio_root"):
        if config.get(name):
            config[name] = str(
                _corpus_path(corpus_root, config[name], field=name, environ=env)
            )

    adapter_value = config.get("adapter_root")
    if adapter_value:
        adapter_root = _repo_path(
            adapter_value, field="adapter_root", root=repo, environ=env
        )
        config["adapter_root"] = str(adapter_root)
        for name in ("corpus_manifest", "alignment_root"):
            if config.get(name):
                config[name] = str(
                    _corpus_path(adapter_root, config[name], field=name, environ=env)
                )
    else:
        for name in ("corpus_manifest", "alignment_root"):
            if config.get(name):
                config[name] = str(
                    _corpus_path(corpus_root, config[name], field=name, environ=env)
                )

    for name, default in (
        ("output_root", "embeddings"),
        ("embedding_store_root", "embedding_store"),
        ("analysis_root", "analysis"),
        ("log_root", "logs"),
        ("temporary_root", "embeddings/.tmp"),
        ("vendor_root", "vendor/clamp3"),
        ("checkpoint_path", "models/clamp3-c2/checkpoint.pth"),
        ("saas_checkpoint_path", "models/clamp3-saas/checkpoint.pth"),
        ("prompt_bank", "prompts/prompt_bank_v1.csv"),
    ):
        config[name] = str(
            _repo_path(config.get(name, default), field=name, root=repo, environ=env)
        )

    requested_device = device or str(config.get("device", "auto"))
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise PipelineConfigurationError("device must be auto, cpu, or cuda")
    config["device"] = requested_device
    config["repository_root"] = str(repo)
    config["configuration_path"] = str(config_path)
    config.setdefault("normalize_embeddings", True)
    config.setdefault("random_seed", 20260819)
    config.setdefault("batch_size_text", 32)
    config.setdefault("batch_size_symbolic", 1)
    config.setdefault("batch_size_audio", 1)
    config.setdefault("num_workers", 0)
    return config


def resolve_corpus_asset(corpus_root: str | Path, relative: str) -> Path:
    """Resolve a non-empty manifest path without allowing traversal."""

    root = Path(corpus_root).resolve()
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise CorpusManifestError(f"corpus asset path must be repository-relative: {relative}")
    path = (root / rel).resolve()
    if path != root and root not in path.parents:
        raise CorpusManifestError(f"corpus asset path escapes corpus_root: {relative}")
    return path


def read_corpus_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest = Path(path)
    if not manifest.is_file():
        raise CorpusManifestError(f"corpus manifest does not exist: {manifest}")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = [name for name in CORPUS_MANIFEST_FIELDS if name not in columns]
        if missing:
            raise CorpusManifestError(
                f"corpus manifest is missing columns: {', '.join(missing)}"
            )
        rows = [
            {name: "" if row.get(name) is None else str(row[name]).strip() for name in reader.fieldnames or []}
            for row in reader
        ]
    if not rows:
        raise CorpusManifestError("corpus manifest contains no rows")
    seen_passages: set[str] = set()
    for number, row in enumerate(rows, start=2):
        passage_id = row["passage_id"]
        if not passage_id or not _SAFE_ID.fullmatch(passage_id):
            raise CorpusManifestError(f"row {number}: invalid passage_id {passage_id!r}")
        if passage_id in seen_passages:
            raise CorpusManifestError(f"row {number}: duplicate passage_id {passage_id!r}")
        seen_passages.add(passage_id)
        try:
            start, end = int(row["bar_start"]), int(row["bar_end"])
        except ValueError as exc:
            raise CorpusManifestError(f"row {number}: invalid bar range") from exc
        if start < 1 or end < start:
            raise CorpusManifestError(f"row {number}: invalid bar range {start}-{end}")
    require_consistent_registrations(
        rows,
        key_field="recording_id",
        stable_fields=RECORDING_REGISTRATION_FIELDS,
        noun="recording_id",
    )
    return rows


def _ascii_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return "".join(char for char in normalized if not unicodedata.combining(char)).casefold()


def composer_matches(value: str, requested: str | None) -> bool:
    if not requested:
        return True
    return _ascii_key(requested) in _ascii_key(value)


def collect_registration_conflicts(
    rows: Iterable[Mapping[str, str]],
    *,
    key_field: str,
    stable_fields: Iterable[str],
) -> list[dict[str, Any]]:
    """Return repeated registrations whose stable metadata disagree."""

    grouped: dict[str, list[Mapping[str, str]]] = {}
    for row in rows:
        key = _text(row.get(key_field, ""))
        if key:
            grouped.setdefault(key, []).append(row)
    conflicts: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        mismatches: list[dict[str, Any]] = []
        for field in stable_fields:
            values = sorted({_text(row.get(field, "")) for row in group})
            if len(values) > 1:
                mismatches.append({"field": field, "values": values})
        if mismatches:
            conflicts.append(
                {
                    "key_field": key_field,
                    "key": key,
                    "row_count": len(group),
                    "mismatches": mismatches,
                }
            )
    return conflicts


def format_registration_conflicts(
    conflicts: Iterable[Mapping[str, Any]],
    *,
    noun: str,
) -> list[str]:
    messages: list[str] = []
    for conflict in conflicts:
        fields = ", ".join(
            f"{item['field']}={item['values']}"
            for item in conflict.get("mismatches", ())
            if isinstance(item, Mapping)
        )
        messages.append(f"{noun} {_text(conflict.get('key', ''))!r} has conflicting registrations: {fields}")
    return messages


def require_consistent_registrations(
    rows: Iterable[Mapping[str, str]],
    *,
    key_field: str,
    stable_fields: Iterable[str],
    noun: str,
) -> None:
    conflicts = collect_registration_conflicts(
        rows,
        key_field=key_field,
        stable_fields=stable_fields,
    )
    if conflicts:
        raise CorpusManifestError(" ; ".join(format_registration_conflicts(conflicts, noun=noun)))


def filter_manifest_rows(
    rows: Iterable[dict[str, str]],
    *,
    composer: str | None = None,
    work_id: str | None = None,
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if composer_matches(row.get("composer", ""), composer)
        and (not work_id or row.get("work_id") == work_id)
    ]


def read_lookup(path: str | Path, key: str) -> dict[str, dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        return {}
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row.get(key, "")): dict(row) for row in rows if row.get(key)}


def select_symbolic_records(
    rows: Iterable[dict[str, str]], corpus_root: str | Path
) -> list[dict[str, Any]]:
    """Return one deterministic record per movement, retaining unsupported assets."""

    row_list = list(rows)
    require_consistent_registrations(
        row_list,
        key_field="movement_id",
        stable_fields=SYMBOLIC_REGISTRATION_FIELDS,
        noun="movement_id",
    )
    root = Path(corpus_root)
    works = read_lookup(root / "manifests" / "works.csv", "work_id")
    movements = read_lookup(root / "manifests" / "movements.csv", "movement_id")
    chosen: dict[str, dict[str, Any]] = {}
    for row in sorted(
        row_list,
        key=lambda item: (item["composer"], item["work_id"], item["movement_id"], item["passage_id"]),
    ):
        movement_id = row["movement_id"]
        if movement_id in chosen:
            continue
        movement = movements.get(movement_id, {})
        work = works.get(row["work_id"], {})
        candidates = [
            row.get("mxl_path", ""),
            movement.get("mxl_path", ""),
            row.get("score_path", ""),
            movement.get("score_path", ""),
            row.get("midi_path", ""),
            movement.get("midi_path", ""),
        ]
        unique_candidates = list(dict.fromkeys(value for value in candidates if value))
        supported = next(
            (value for value in unique_candidates if Path(value).suffix.lower() in SYMBOLIC_EXTENSIONS),
            "",
        )
        original = supported or (unique_candidates[0] if unique_candidates else "")
        source = resolve_corpus_asset(root, original) if original else None
        if not original:
            status, error = "unavailable", "manifest records no symbolic source"
        elif Path(original).suffix.lower() not in SYMBOLIC_EXTENSIONS:
            status, error = (
                "unsupported",
                f"CLaMP supports MusicXML/MXL/MIDI, not {Path(original).suffix or row.get('score_format')}",
            )
        elif not source or not source.is_file():
            status, error = "unavailable", "symbolic source is not present locally"
        else:
            status, error = "pending", ""
        chosen[movement_id] = {
            "score_id": movement_id,
            "composer": row["composer"],
            "period": row["period"],
            "composition_id": row.get("composition_id", ""),
            "work_id": row["work_id"],
            "movement_id": movement_id,
            "title": work.get("title") or movement.get("movement_title") or row.get("work_title") or row["work_id"],
            "source_path": original,
            "resolved_source_path": source,
            "source_format": Path(original).suffix.lower().lstrip(".") if original else row.get("score_format", ""),
            "status": status,
            "error_message": error,
        }
    return list(chosen.values())


def select_recording_records(
    rows: Iterable[dict[str, str]], corpus_root: str | Path
) -> list[dict[str, Any]]:
    """Return one record per declared recording, with performer and rights metadata."""

    row_list = list(rows)
    require_consistent_registrations(
        row_list,
        key_field="recording_id",
        stable_fields=RECORDING_REGISTRATION_FIELDS,
        noun="recording_id",
    )
    root = Path(corpus_root)
    lookup = read_lookup(root / "manifests" / "recordings.csv", "recording_id")
    chosen: dict[str, dict[str, Any]] = {}
    for row in sorted(
        row_list,
        key=lambda item: (item["composer"], item["work_id"], item["movement_id"], item["recording_id"]),
    ):
        recording_id = row.get("recording_id", "")
        if not recording_id or recording_id in chosen:
            continue
        detail = lookup.get(recording_id, {})
        audio_path = row.get("audio_path") or detail.get("audio_path", "")
        source = resolve_corpus_asset(root, audio_path) if audio_path else None
        if not audio_path:
            status, error = "unavailable", "manifest records no local audio_path"
        elif Path(audio_path).suffix.lower() not in AUDIO_EXTENSIONS:
            status, error = "unsupported", f"unsupported audio format {Path(audio_path).suffix}"
        elif not source or not source.is_file():
            status, error = "unavailable", "audio_path is not present locally"
        else:
            status, error = "pending", ""
        chosen[recording_id] = {
            "recording_id": recording_id,
            "composer": row["composer"],
            "period": row["period"],
            "composition_id": row.get("composition_id", ""),
            "work_id": row["work_id"],
            "movement_id": row["movement_id"],
            "performer": detail.get("performer", ""),
            "audio_path": audio_path,
            "resolved_audio_path": source,
            "rights_status": row.get("rights_status") or detail.get("rights_status", ""),
            "audio_origin": row.get("audio_origin", ""),
            "declared_duration_seconds": detail.get("duration_seconds", ""),
            "status": status,
            "error_message": error,
        }
    return list(chosen.values())


def parse_bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y", "on"}


def load_prompt_bank(path: str | Path, *, enabled_only: bool = True) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise PromptBankError(f"prompt bank does not exist: {source}")
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = [name for name in PROMPT_FIELDS if name not in columns]
        if missing:
            raise PromptBankError(f"prompt bank is missing columns: {', '.join(missing)}")
        rows = [{name: str(row.get(name, "")) for name in PROMPT_FIELDS} for row in reader]
    seen: set[str] = set()
    selected: list[dict[str, str]] = []
    for number, row in enumerate(rows, start=2):
        prompt_id = row["prompt_id"].strip()
        text = row["prompt_text"]
        if not prompt_id or not _SAFE_ID.fullmatch(prompt_id):
            raise PromptBankError(f"row {number}: invalid prompt_id {prompt_id!r}")
        if prompt_id in seen:
            raise PromptBankError(f"row {number}: duplicate prompt_id {prompt_id!r}")
        seen.add(prompt_id)
        if not text.strip():
            raise PromptBankError(f"row {number}: prompt_text is empty")
        if not enabled_only or parse_bool(row["enabled"]):
            selected.append(row)
    if enabled_only and not selected:
        raise PromptBankError("prompt bank has no enabled rows")
    return selected
