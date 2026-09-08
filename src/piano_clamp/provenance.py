"""Immutable input fingerprints for safe, resumable experiment runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


class FingerprintError(RuntimeError):
    """Raised when an existing run was created from different inputs."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(value: Any) -> str:
    """Hash a JSON-compatible value using stable key and separator settings."""

    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_data_payload(
    rows: Sequence[Mapping[str, str]], sources: Mapping[str, Path]
) -> dict[str, Any]:
    """Describe manifest metadata and exact source bytes."""

    files = []
    for row in rows:
        passage_id = row["passage_id"]
        source = sources[passage_id]
        files.append(
            {
                "passage_id": passage_id,
                "relative_path": row["relative_path"],
                "size_bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        )
    return {
        "manifest_rows": [dict(row) for row in rows],
        "source_files": files,
    }


def build_model_payload(config: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    """Describe a configured checkpoint without repeatedly hashing multi-GB weights."""

    model = config["models"][model_name]
    checkpoint = Path(model["checkpoint"])
    return {
        "model": model_name,
        "modality": model.get("modality"),
        "checkpoint_name": checkpoint.name,
        "checkpoint_size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
        "checkpoint_sha256": model.get("sha256"),
        "clamp3_upstream_commit": config.get("clamp3", {}).get("upstream_commit"),
    }


def build_prompt_payload(prompts: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    """Describe exact prompt wording and preregistration metadata."""

    return {"prompts": [dict(prompt) for prompt in prompts]}


def ensure_input_lock(
    run_dir: str | Path,
    name: str,
    payload: Mapping[str, Any],
) -> Path:
    """Create an input lock, or reject reuse when the current payload differs."""

    root = Path(run_dir) / "inputs"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.lock.json"
    document = {
        "schema_version": 1,
        "name": name,
        "fingerprint": canonical_fingerprint(payload),
        "payload": payload,
    }
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FingerprintError(f"input lock is unreadable: {path}") from exc
        if previous.get("fingerprint") != document["fingerprint"]:
            raise FingerprintError(
                f"inputs for existing run have changed ({name}); choose a new experiment.run_name "
                f"instead of reusing {Path(run_dir).name!r}. Existing lock: {path}"
            )
        return path

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path

