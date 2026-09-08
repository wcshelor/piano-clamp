"""Prompt configuration parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .paths import load_yaml


class PromptError(ValueError):
    """Raised for malformed or duplicate prompt definitions."""


def load_prompts(paths: Iterable[str | Path]) -> list[dict[str, str]]:
    """Load ordered prompt records from one or more YAML files."""

    prompts: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        document = load_yaml(path)
        values = document.get("prompts")
        if not isinstance(values, list) or not values:
            raise PromptError(f"prompt file has no non-empty 'prompts' list: {path}")
        for index, value in enumerate(values, start=1):
            if not isinstance(value, dict):
                raise PromptError(f"{path}: prompt {index} must be a mapping")
            prompt_id = str(value.get("id", "")).strip()
            text = str(value.get("text", "")).strip()
            if not prompt_id or not text:
                raise PromptError(f"{path}: prompt {index} requires non-empty id and text")
            if prompt_id in seen:
                raise PromptError(f"duplicate prompt id: {prompt_id}")
            compact_id = prompt_id.replace("-", "").replace(".", "").replace("_", "")
            if Path(prompt_id).name != prompt_id or not compact_id.isalnum():
                raise PromptError(f"prompt id is not filename-safe: {prompt_id!r}")
            seen.add(prompt_id)
            prompts.append({"prompt_id": prompt_id, "prompt_text": text})
    return prompts
