"""Reproducibility metadata for every embedding stage."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .embedding_io import content_hash, sha256_path


def make_run_id(stage: str, config: Mapping[str, Any], *, now: datetime | None = None) -> str:
    timestamp = now or datetime.now(timezone.utc)
    portable_config = portable_configuration(config)
    return f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{content_hash([stage, portable_config])[:12]}"


def stage_log_path(
    config: Mapping[str, Any],
    stem: str,
    *,
    run_id: str,
) -> Path:
    return Path(str(config["log_root"])) / f"{stem}-{run_id}.log"


def git_commit(path: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def portable_path(path: str | Path, config: Mapping[str, Any]) -> str:
    """Represent paths relative to known roots before falling back to a name."""

    target = Path(path).expanduser().resolve()
    for key, label in (("repository_root", "${REPOSITORY_ROOT}"), ("corpus_root", "${CORPUS_ROOT}")):
        if not config.get(key):
            continue
        root = Path(str(config[key])).resolve()
        try:
            relative = target.relative_to(root)
        except ValueError:
            continue
        return label if str(relative) == "." else f"{label}/{relative.as_posix()}"
    return target.name


def portable_configuration(config: Mapping[str, Any]) -> dict[str, Any]:
    path_keys = {
        "repository_root",
        "corpus_root",
        "corpus_manifest",
        "score_root",
        "audio_root",
        "alignment_root",
        "output_root",
        "embedding_store_root",
        "analysis_root",
        "log_root",
        "temporary_root",
        "vendor_root",
        "checkpoint_path",
        "saas_checkpoint_path",
        "prompt_bank",
        "configuration_path",
    }
    portable: dict[str, Any] = {}
    for key, value in config.items():
        if key in path_keys and value:
            portable[key] = portable_path(str(value), config)
        else:
            portable[key] = value
    return portable


def runtime_environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "PyYAML", "torch", "transformers", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "unavailable"
    torch_version = packages["torch"]
    cuda_version: str | None = None
    cuda_available = False
    try:
        import torch

        cuda_version = torch.version.cuda
        cuda_available = bool(torch.cuda.is_available())
    except ImportError:
        pass
    return {
        "python_version": platform.python_version(),
        "python_executable": Path(sys.executable).name,
        "platform": platform.platform(),
        "torch_version": torch_version,
        "cuda_version": cuda_version,
        "cuda_available": cuda_available,
        "packages": packages,
    }


def build_run_metadata(
    config: Mapping[str, Any],
    *,
    stage: str,
    checkpoint_path: str | Path,
    model_space: str,
    run_id: str | None = None,
    embedding_dimension: int = 768,
    batch_size: int = 1,
    input_manifest_path: str | Path | None = None,
    prompt_bank_path: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    checkpoint = Path(checkpoint_path)
    configured_hash = (
        config.get(f"{model_space}_checkpoint_sha256")
        or (config.get("checkpoint_sha256") if model_space == "c2" else None)
    )
    checkpoint_hash = str(configured_hash or sha256_path(checkpoint)) if checkpoint.is_file() else "unavailable"
    manifest_hash = (
        sha256_path(input_manifest_path)
        if input_manifest_path and Path(input_manifest_path).is_file()
        else None
    )
    prompt_hash = (
        sha256_path(prompt_bank_path)
        if prompt_bank_path and Path(prompt_bank_path).is_file()
        else None
    )
    portable_config = portable_configuration(config)
    resolved_run_id = run_id or make_run_id(stage, config, now=now)
    vendor = Path(str(config["vendor_root"]))
    environment = runtime_environment()
    document: dict[str, Any] = {
        "run_id": resolved_run_id,
        "timestamp": now.isoformat(),
        "hostname": socket.gethostname(),
        "stage": stage,
        "model": "CLaMP 3",
        "model_space": model_space,
        "python_version": environment["python_version"],
        "torch_version": environment["torch_version"],
        "cuda_version": environment["cuda_version"],
        "device": config.get("resolved_device", config.get("device", "auto")),
        "CLaMP_repository_commit": git_commit(vendor),
        "piano_clamp_repository_commit": git_commit(config["repository_root"]),
        "checkpoint_path": portable_path(checkpoint, config),
        "checkpoint_hash": checkpoint_hash,
        "configuration_hash": content_hash(portable_config),
        "configuration": portable_config,
        "prompt_bank_hash": prompt_hash,
        "input_manifest_hash": manifest_hash,
        "embedding_dimension": embedding_dimension,
        "normalization": "l2" if config.get("normalize_embeddings", True) else "none",
        "batch_size": int(batch_size),
        "random_seed": int(config.get("random_seed", 20260819)),
        "environment": environment,
    }
    if extra:
        document.update(dict(extra))
    return document
