#!/usr/bin/env python3
"""Write a committed-friendly inventory of heavyweight HPC embedding artifacts.

The HPC creates large files that this repository deliberately does not track:
embedding matrices, stage CSVs, immutable embedding-store snapshots, and other
derived artifacts. That makes normal git history insufficient for answering
"what embeddings already exist on the cluster?" This script bridges that gap by
scanning the actual HPC artifact roots and writing small report files that are
safe to commit.

Run this after any command or Slurm job that may create, replace, delete, or
snapshot embedding bundles. When giving a user HPC terminal commands for such a
job, include this script as the final step so `reports/hpc/artifact_inventory/`
stays current.

Typical HPC usage:

    python scripts/inventory_hpc_embeddings.py \
      --embedding-root /home/student/w/wshelor/share/piano-clamp-runs/embeddings \
      --store-root /share/users/student/w/wshelor/piano-clamp/embedding_store \
      --output-dir reports/hpc/artifact_inventory

Outputs:

    embeddings_inventory.csv  full table for agents and scripts
    embeddings_inventory.md   human-readable summary
    latest.json               summary plus full artifact records

The inventory records only metadata, paths, counts, shapes, and hashes. It does
not copy or commit raw embedding matrices.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.data_loading import load_pipeline_config  # noqa: E402
from piano_clamp.embedding_io import sha256_path  # noqa: E402


FIELDS = (
    "artifact_id",
    "location",
    "kind",
    "status",
    "path",
    "snapshot_path",
    "created_at",
    "updated_at",
    "git_commit",
    "stage",
    "model",
    "model_space",
    "checkpoint_path",
    "checkpoint_hash",
    "configuration_hash",
    "device",
    "row_count",
    "metadata_record_count",
    "failure_count",
    "embedding_shape",
    "embedding_dimension",
    "matrix_file",
    "table_file",
    "matrix_sha256",
    "table_sha256",
    "metadata_sha256",
    "source_material",
    "composer_scope",
    "window_modes",
    "overlap",
    "run_id",
    "notes",
)


# Live bundle locations relative to the configured or explicitly supplied
# embedding root. Keep this list aligned with the bundle locations produced by
# the embedding scripts and validated by `scripts/validate_embeddings.py`.
LIVE_BUNDLES = (
    "text",
    "text_saas",
    "symbolic",
    "audio",
    "passages",
    "passages/performance_midi",
    "passages/audio",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _csv_rows_and_values(path: Path) -> tuple[int, Counter[str], set[str], set[str]]:
    composers: Counter[str] = Counter()
    methods: set[str] = set()
    materials: set[str] = set()
    rows = 0
    if not path.is_file():
        return rows, composers, methods, materials
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows += 1
            composer = str(row.get("composer", "")).strip()
            if composer:
                composers[composer] += 1
            method = str(row.get("passage_generation_method", "")).strip()
            if method:
                methods.add(method)
            material = str(row.get("source_material", "") or row.get("embedding_modality", "")).strip()
            if material:
                materials.add(material)
    return rows, composers, methods, materials


def _shape(path: Path) -> str:
    if not path.is_file():
        return ""
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return "x".join(str(part) for part in array.shape)
    except Exception:
        return ""


def _mtime(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return ""


def _hash_if_exists(path: Path) -> str:
    return sha256_path(path) if path.is_file() else ""


def _kind(metadata: dict[str, Any], directory: Path) -> str:
    stage = str(metadata.get("stage", "")).strip()
    if stage:
        return stage
    name = directory.name
    if name == "text":
        return "text"
    if name == "symbolic":
        return "symbolic"
    if name == "performance_midi":
        return "passages_performance_midi"
    if name == "audio" and directory.parent.name == "passages":
        return "passages_audio"
    if name == "passages":
        return "passages_symbolic"
    return name or "unknown"


def _window_modes(methods: set[str]) -> str:
    values: set[str] = set()
    for method in methods:
        prefix = "fixed_"
        suffix = "_bars"
        if method.startswith(prefix) and method.endswith(suffix):
            values.add(method[len(prefix) : -len(suffix)])
        else:
            values.add(method)
    return ",".join(sorted(values, key=lambda value: (len(value), value)))


def _composer_scope(composers: Counter[str]) -> str:
    if not composers:
        return ""
    return "; ".join(f"{name}={count}" for name, count in sorted(composers.items()))


def _artifact_id(location: str, directory: Path, metadata: dict[str, Any]) -> str:
    run_id = str(metadata.get("run_id", "")).strip()
    stage = str(metadata.get("stage", "")).strip()
    payload = json.dumps(
        {
            "location": location,
            "path": str(directory),
            "run_id": run_id,
            "stage": stage,
            "matrix_sha256": str(metadata.get("embedding_sha256", "")),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def inspect_bundle(directory: Path, *, location: str, snapshot_path: str = "") -> dict[str, str] | None:
    metadata_path = directory / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = _read_json(metadata_path)
    matrix_file = str(metadata.get("matrix_file", ""))
    table_file = str(metadata.get("table_file", ""))
    matrix_path = directory / matrix_file if matrix_file else Path()
    table_path = directory / table_file if table_file else Path()
    table_rows, composers, methods, materials = _csv_rows_and_values(table_path)
    shape = _shape(matrix_path)
    notes: list[str] = []
    if not matrix_path.is_file():
        notes.append("missing matrix file")
    if not table_path.is_file():
        notes.append("missing table file")
    if table_rows and metadata.get("metadata_record_count") not in {"", None, table_rows}:
        notes.append("table row count differs from metadata_record_count")
    source_material = ";".join(sorted(materials))
    if not source_material:
        source_material = str(metadata.get("source_material", ""))
    matrix_sha = str(metadata.get("embedding_sha256", "")) or _hash_if_exists(matrix_path)
    return {
        "artifact_id": _artifact_id(location, directory, metadata),
        "location": location,
        "kind": _kind(metadata, directory),
        "status": "current" if location == "live" else "snapshot",
        "path": str(directory),
        "snapshot_path": snapshot_path,
        "created_at": str(metadata.get("timestamp", "")),
        "updated_at": _mtime(metadata_path),
        "git_commit": str(metadata.get("git_commit", "")),
        "stage": str(metadata.get("stage", "")),
        "model": str(metadata.get("model", "")),
        "model_space": str(metadata.get("model_space", "")),
        "checkpoint_path": str(metadata.get("checkpoint_path", "")),
        "checkpoint_hash": str(metadata.get("checkpoint_hash", "")),
        "configuration_hash": str(metadata.get("configuration_hash", "")),
        "device": str(metadata.get("device", "")),
        "row_count": str(metadata.get("row_count", "")),
        "metadata_record_count": str(metadata.get("metadata_record_count", table_rows or "")),
        "failure_count": str(metadata.get("failure_count", "")),
        "embedding_shape": shape or "x".join(str(part) for part in metadata.get("embedding_shape", [])),
        "embedding_dimension": str(metadata.get("embedding_dimension", "")),
        "matrix_file": matrix_file,
        "table_file": table_file,
        "matrix_sha256": matrix_sha,
        "table_sha256": _hash_if_exists(table_path),
        "metadata_sha256": _hash_if_exists(metadata_path),
        "source_material": source_material,
        "composer_scope": _composer_scope(composers),
        "window_modes": _window_modes(methods),
        "overlap": "overlap-half" if any("overlap" in method for method in methods) else "",
        "run_id": str(metadata.get("run_id", "")),
        "notes": "; ".join(notes),
    }


def _configured_roots(config_path: str) -> tuple[list[Path], list[Path]]:
    try:
        config = load_pipeline_config(config_path)
    except Exception:
        return [], []
    embedding_root = Path(str(config.get("output_root", ""))).expanduser()
    store_root = Path(str(config.get("embedding_store_root", ""))).expanduser()
    embedding_roots = [embedding_root] if str(embedding_root) else []
    store_roots = [store_root] if str(store_root) else []
    return embedding_roots, store_roots


def discover_live_bundles(root: Path) -> list[Path]:
    return [root / relative for relative in LIVE_BUNDLES if (root / relative / "metadata.json").is_file()]


def discover_store_bundles(root: Path) -> list[Path]:
    runs = root / "runs"
    if not runs.is_dir():
        return []
    return sorted(path.parent for path in runs.rglob("metadata.json"))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any], rows: list[dict[str, str]]) -> None:
    live = [row for row in rows if row["location"] == "live"]
    lines = [
        "# HPC Embedding Artifact Inventory",
        "",
        f"Generated: {payload['generated_at']}",
        f"Host: {payload['host']}",
        f"Git commit: {payload['git_commit'] or 'unknown'}",
        "",
        "## Current Live Bundles",
        "",
        "| Kind | Path | Rows | Shape | Model | Device | Updated | Status | Notes |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    if live:
        for row in live:
            lines.append(
                "| {kind} | `{path}` | {rows} | {shape} | {model} {space} | {device} | {updated} | {status} | {notes} |".format(
                    kind=row["kind"],
                    path=row["path"],
                    rows=row["row_count"] or row["metadata_record_count"],
                    shape=row["embedding_shape"],
                    model=row["model"] or "model",
                    space=row["model_space"],
                    device=row["device"],
                    updated=row["updated_at"],
                    status=row["status"],
                    notes=row["notes"],
                )
            )
    else:
        lines.append("| none found |  |  |  |  |  |  | missing |  |")
    lines.extend(
        [
            "",
            "## Snapshot Summary",
            "",
            f"- Total artifacts: {len(rows)}",
            f"- Live artifacts: {sum(row['location'] == 'live' for row in rows)}",
            f"- Store snapshots: {sum(row['location'] == 'store' for row in rows)}",
            "",
            "## Files",
            "",
            "- `embeddings_inventory.csv`: full machine-readable artifact table.",
            "- `latest.json`: compact summary and root paths used for this inventory.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument("--embedding-root", action="append", type=Path)
    parser.add_argument("--store-root", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/hpc/artifact_inventory"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configured_embedding_roots, configured_store_roots = _configured_roots(args.config)
    embedding_roots = args.embedding_root or configured_embedding_roots
    store_roots = args.store_root or configured_store_roots
    rows: list[dict[str, str]] = []
    missing_roots: list[str] = []

    for root in embedding_roots:
        root = root.expanduser()
        if not root.exists():
            missing_roots.append(str(root))
            continue
        for bundle in discover_live_bundles(root):
            row = inspect_bundle(bundle, location="live")
            if row:
                rows.append(row)

    for root in store_roots:
        root = root.expanduser()
        if not root.exists():
            missing_roots.append(str(root))
            continue
        for bundle in discover_store_bundles(root):
            row = inspect_bundle(bundle, location="store", snapshot_path=str(bundle))
            if row:
                rows.append(row)

    rows.sort(key=lambda row: (row["location"], row["kind"], row["updated_at"], row["path"]))
    summary = {
        "generated_at": _now(),
        "host": socket.gethostname(),
        "git_commit": _git_head(),
        "embedding_roots": [str(path) for path in embedding_roots],
        "store_roots": [str(path) for path in store_roots],
        "missing_roots": missing_roots,
        "artifact_count": len(rows),
        "live_count": sum(row["location"] == "live" for row in rows),
        "store_count": sum(row["location"] == "store" for row in rows),
        "kinds": dict(sorted(Counter(row["kind"] for row in rows).items())),
    }

    output_dir = args.output_dir
    write_csv(output_dir / "embeddings_inventory.csv", rows)
    write_json(output_dir / "latest.json", {**summary, "artifacts": rows})
    write_markdown(output_dir / "embeddings_inventory.md", summary, rows)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
