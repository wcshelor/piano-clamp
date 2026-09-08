"""Configuration loading and safe external path resolution."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigurationError(ValueError):
    """Raised when experiment configuration is incomplete or unsafe."""


_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


def repository_root() -> Path:
    """Return the repository root for an editable or source-tree install."""

    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"YAML file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Expected a YAML mapping in {config_path}")
    return value


def _environment_path(value: Any, *, field: str, environ: Mapping[str, str]) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a non-empty path string")
    match = _ENV_REFERENCE.fullmatch(value.strip())
    if match:
        variable = match.group(1)
        resolved = environ.get(variable)
        if not resolved:
            raise ConfigurationError(
                f"{field} references ${variable}, but {variable} is not set"
            )
        value = resolved
    else:
        value = os.path.expandvars(value)
        if "$" in value:
            raise ConfigurationError(f"{field} contains an unresolved environment variable")
    return Path(value).expanduser().resolve()


def path_within(root: str | Path, relative: str | Path, *, field: str) -> Path:
    """Resolve a relative path below *root*, rejecting traversal and absolutes."""

    root_path = Path(root).expanduser().resolve()
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ConfigurationError(f"{field} must be relative to {root_path}")
    candidate = (root_path / relative_path).resolve()
    if candidate != root_path and root_path not in candidate.parents:
        raise ConfigurationError(f"{field} escapes its configured root: {relative}")
    return candidate


def load_resolved_config(
    config_path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load a config and resolve repository and external-root paths.

    External data and run artifacts are rooted only through MUSIC_DATA_ROOT and
    CLAMP3_RUN_ROOT (normally referenced in the YAML as ``${...}``).
    """

    environment = os.environ if environ is None else environ
    root = Path(repo_root).resolve() if repo_root else repository_root()
    raw = load_yaml(config_path)
    config = copy.deepcopy(raw)

    experiment = config.get("experiment")
    if not isinstance(experiment, dict) or not experiment.get("run_name"):
        raise ConfigurationError("experiment.run_name is required")
    run_name = str(experiment["run_name"])
    if Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ConfigurationError("experiment.run_name must be one safe path component")

    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ConfigurationError("paths must be a YAML mapping")

    data_root = _environment_path(
        paths.get("data_root", "${MUSIC_DATA_ROOT}"),
        field="paths.data_root",
        environ=environment,
    )
    run_root = _environment_path(
        paths.get("run_root", "${CLAMP3_RUN_ROOT}"),
        field="paths.run_root",
        environ=environment,
    )
    paths["data_root"] = str(data_root)
    paths["run_root"] = str(run_root)
    run_dir = path_within(run_root, run_name, field="experiment.run_name")
    if (
        run_dir == data_root
        or data_root in run_dir.parents
        or run_dir in data_root.parents
    ):
        raise ConfigurationError(
            "the configured run directory and MUSIC_DATA_ROOT must not overlap"
        )
    paths["run_dir"] = str(run_dir)

    manifest_value = paths.get("manifest")
    if not manifest_value:
        raise ConfigurationError("paths.manifest is required")
    paths["manifest"] = str(path_within(data_root, manifest_value, field="paths.manifest"))
    if paths.get("audio_manifest"):
        paths["audio_manifest"] = str(
            path_within(data_root, paths["audio_manifest"], field="paths.audio_manifest")
        )

    models = config.get("models")
    if not isinstance(models, dict):
        raise ConfigurationError("models must be a YAML mapping")
    for model_name in ("c2", "saas"):
        model = models.get(model_name)
        if not isinstance(model, dict) or not model.get("checkpoint"):
            raise ConfigurationError(f"models.{model_name}.checkpoint is required")
        checkpoint = Path(str(model["checkpoint"]))
        model["checkpoint"] = str(
            checkpoint.resolve() if checkpoint.is_absolute() else (root / checkpoint).resolve()
        )

    prompt_files = config.get("prompt_files")
    if not isinstance(prompt_files, list) or not prompt_files:
        raise ConfigurationError("prompt_files must be a non-empty list")
    config["prompt_files"] = [
        str((root / Path(str(item))).resolve()) if not Path(str(item)).is_absolute() else str(Path(str(item)).resolve())
        for item in prompt_files
    ]
    paths["repository_root"] = str(root)
    return config
