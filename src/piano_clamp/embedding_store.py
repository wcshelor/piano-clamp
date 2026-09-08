"""Append-only storage for immutable embedding bundle snapshots."""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .embedding_io import content_hash, read_embedding_bundle, sha256_path


class EmbeddingStoreError(RuntimeError):
    """Raised when an embedding snapshot cannot be stored safely."""


REGISTRY_FIELDS = (
    "registry_id",
    "identity_key",
    "run_id",
    "timestamp",
    "stage",
    "item_id",
    "item_type",
    "model_space",
    "checkpoint_hash",
    "configuration_hash",
    "bundle_name",
    "bundle_path",
    "embedding_row",
    "embedding_dimension",
    "status",
    "composer",
    "period",
    "composition_id",
    "work_id",
    "movement_id",
    "recording_id",
    "performer",
    "audio_origin",
    "rights_status",
    "passage_generation_method",
    "window_bars",
    "bar_start",
    "bar_end",
    "embedding_modality",
    "source_material",
    "source_path",
    "score_path",
    "audio_path",
    "midi_path",
    "performance_midi_path",
    "content_hash",
    "matrix_sha256",
)


def _bundle_name(path: Path) -> str:
    parts = [part for part in path.parts if part not in {".", ""}]
    if not parts:
        return "bundle"
    return "__".join(parts)


def _window_bars(value: str) -> str:
    prefix = "fixed_"
    suffix = "_bars"
    if value.startswith(prefix) and value.endswith(suffix):
        return value[len(prefix) : -len(suffix)]
    return ""


def _item_type(metadata: Mapping[str, Any], row: Mapping[str, str], relative_bundle: Path) -> str:
    stage = str(metadata.get("stage", ""))
    if stage == "symbolic":
        return "score"
    if stage == "audio":
        return "recording_audio"
    if stage == "audio_chunks":
        return "audio_chunk"
    if stage == "text":
        return "text_prompt"
    if stage.startswith("passages_"):
        modality = str(row.get("embedding_modality", "")) or stage.split("_", 1)[1]
        return f"passage_{modality}"
    if "audio" in relative_bundle.parts:
        return "audio"
    return "unknown"


def _identity_key(
    metadata: Mapping[str, Any],
    row: Mapping[str, str],
    *,
    item_id: str,
    item_type: str,
) -> str:
    payload = {
        "item_id": item_id,
        "item_type": item_type,
        "model_space": str(metadata.get("model_space", "")),
        "checkpoint_hash": str(metadata.get("checkpoint_hash", "")),
        "configuration_hash": str(metadata.get("configuration_hash", "")),
        "content_hash": str(row.get("content_hash", "")),
        "recording_id": str(row.get("recording_id", "")),
        "movement_id": str(row.get("movement_id", "")),
        "passage_generation_method": str(row.get("passage_generation_method", "")),
        "bar_start": str(row.get("bar_start", "")),
        "bar_end": str(row.get("bar_end", "")),
        "embedding_modality": str(row.get("embedding_modality", "")),
        "source_material": str(row.get("source_material", "")),
    }
    return content_hash(payload)


def _registry_row(
    metadata: Mapping[str, Any],
    row: Mapping[str, str],
    *,
    relative_bundle: Path,
    matrix_sha256: str,
) -> dict[str, str]:
    id_field = str(metadata.get("id_field", ""))
    item_id = str(row.get(id_field, ""))
    item_type = _item_type(metadata, row, relative_bundle)
    identity_key = _identity_key(metadata, row, item_id=item_id, item_type=item_type)
    passage_method = str(row.get("passage_generation_method", ""))
    bundle_path = relative_bundle.as_posix() or "."
    material = str(row.get("source_material", "") or row.get("embedding_modality", ""))
    source_path = str(row.get("source_path", ""))
    if not source_path:
        if material == "performance_midi":
            source_path = str(row.get("performance_midi_path", ""))
        elif material == "audio":
            source_path = str(row.get("audio_path", ""))
        else:
            source_path = str(row.get("score_path", "") or row.get("midi_path", ""))
    return {
        "registry_id": content_hash([identity_key, str(row.get("embedding_row", "")), bundle_path]),
        "identity_key": identity_key,
        "run_id": str(metadata.get("run_id", "")),
        "timestamp": str(metadata.get("timestamp", "")),
        "stage": str(metadata.get("stage", "")),
        "item_id": item_id,
        "item_type": item_type,
        "model_space": str(metadata.get("model_space", "")),
        "checkpoint_hash": str(metadata.get("checkpoint_hash", "")),
        "configuration_hash": str(metadata.get("configuration_hash", "")),
        "bundle_name": _bundle_name(relative_bundle),
        "bundle_path": bundle_path,
        "embedding_row": str(row.get("embedding_row", "")),
        "embedding_dimension": str(row.get("embedding_dimension", "")),
        "status": str(row.get("status", "")),
        "composer": str(row.get("composer", "")),
        "period": str(row.get("period", "")),
        "composition_id": str(row.get("composition_id", "")),
        "work_id": str(row.get("work_id", "")),
        "movement_id": str(row.get("movement_id", "")),
        "recording_id": str(row.get("recording_id", "")),
        "performer": str(row.get("performer", "")),
        "audio_origin": str(row.get("audio_origin", "")),
        "rights_status": str(row.get("rights_status", "")),
        "passage_generation_method": passage_method,
        "window_bars": _window_bars(passage_method),
        "bar_start": str(row.get("bar_start", "")),
        "bar_end": str(row.get("bar_end", "")),
        "embedding_modality": str(row.get("embedding_modality", "")),
        "source_material": str(row.get("source_material", "")),
        "source_path": source_path,
        "score_path": str(row.get("score_path", "")),
        "audio_path": str(row.get("audio_path", "")),
        "midi_path": str(row.get("midi_path", "")),
        "performance_midi_path": str(row.get("performance_midi_path", "")),
        "content_hash": str(row.get("content_hash", "")),
        "matrix_sha256": matrix_sha256,
    }


def _read_registry(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_registry(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: str(row.get(field, "")) for field in REGISTRY_FIELDS})


def snapshot_bundle(
    bundle_directory: str | Path,
    *,
    store_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Copy one bundle into immutable storage and append new registry rows."""

    bundle = Path(bundle_directory).resolve()
    output = Path(output_root).resolve()
    root = Path(store_root).resolve()
    matrix, rows, metadata = read_embedding_bundle(bundle)
    try:
        relative_bundle = bundle.relative_to(output)
    except ValueError as exc:
        raise EmbeddingStoreError(f"bundle {bundle} is not inside output_root {output}") from exc
    run_id = str(metadata.get("run_id", "")).strip()
    if not run_id:
        raise EmbeddingStoreError(f"bundle {bundle} metadata.json does not declare run_id")
    matrix_path = bundle / str(metadata.get("matrix_file", ""))
    table_path = bundle / str(metadata.get("table_file", ""))
    metadata_path = bundle / "metadata.json"
    snapshot_dir = root / "runs" / run_id / relative_bundle
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for source in (matrix_path, table_path, metadata_path):
        destination = snapshot_dir / source.name
        if destination.exists():
            if sha256_path(destination) != sha256_path(source):
                raise EmbeddingStoreError(f"snapshot destination already exists with different content: {destination}")
            continue
        shutil.copy2(source, destination)

    registry_path = root / "registry.csv"
    summary_path = root / "registry.json"
    existing = _read_registry(registry_path)
    seen = {row["identity_key"] for row in existing if row.get("identity_key")}
    matrix_sha = sha256_path(snapshot_dir / str(metadata.get("matrix_file", "")))
    appended = 0
    duplicates = 0
    for row in rows:
        entry = _registry_row(metadata, row, relative_bundle=Path("runs") / run_id / relative_bundle, matrix_sha256=matrix_sha)
        if entry["identity_key"] in seen:
            duplicates += 1
            continue
        existing.append(entry)
        seen.add(entry["identity_key"])
        appended += 1
    existing.sort(key=lambda row: (row["run_id"], row["bundle_path"], row["item_type"], row["item_id"]))
    _write_registry(registry_path, existing)
    summary = {
        "row_count": len(existing),
        "successful_count": sum(row.get("status") == "success" for row in existing),
        "failed_count": sum(row.get("status") == "failed" for row in existing),
        "duplicate_skips": duplicates,
        "bundles": sorted({row["bundle_path"] for row in existing if row.get("bundle_path")}),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "snapshot_dir": snapshot_dir,
        "registry_path": registry_path,
        "summary_path": summary_path,
        "appended": appended,
        "duplicates": duplicates,
        "matrix_shape": list(matrix.shape),
    }


def snapshot_written_bundle(paths: Mapping[str, Path], config: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot one newly written bundle using the configured store root."""

    matrix_path = paths.get("matrix")
    if matrix_path is None:
        raise EmbeddingStoreError("bundle paths do not include matrix output")
    return snapshot_bundle(
        Path(matrix_path).parent,
        store_root=str(config["embedding_store_root"]),
        output_root=str(config["output_root"]),
    )
