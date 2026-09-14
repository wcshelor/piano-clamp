"""Command-line interface for the piano-clamp dataset-consumer pipeline.

piano-clamp validates manifests, extracts features, creates embeddings, analyzes
embeddings, and renders review/result artifacts from external datasets. Dataset
creation and source-asset rendering (including audio-from-MIDI) belong outside
this repo, in the shared dataset layer.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .analysis import AnalysisError, analyze, analyze_features
from .data import ManifestError
from .embeddings import (
    EmbeddingError,
    embed_audio,
    embed_prompts,
    embed_scores,
    make_run_manifest,
    score_prompts,
)
from .paths import ConfigurationError, load_resolved_config
from .features import FeatureError, extract_features
from .prompts import PromptError
from .provenance import FingerprintError
from .similarity import SimilarityError
from .validation import DataValidationError, validate_data


COMMANDS: dict[str, Callable[[dict[str, Any]], Path]] = {
    "validate-data": validate_data,
    "extract-features": extract_features,
    "analyze-features": analyze_features,
    "embed-scores": embed_scores,
    "embed-prompts": embed_prompts,
    "score-prompts": score_prompts,
    "embed-audio": embed_audio,
    "analyze": analyze,
    "make-run-manifest": make_run_manifest,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="piano-clamp",
        description="Reproducible CLaMP 3 C2 score/prompt experiments with optional SAAS audio.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    help_text = {
        "validate-data": "validate and summarize external passage data without a GPU",
        "extract-features": "derive transparent features from MusicXML/MXL passages",
        "analyze-features": "analyze transparent features without model embeddings or prompts",
        "embed-scores": "embed manifest scores with CLaMP 3 C2",
        "embed-prompts": "embed configured prompts with CLaMP 3 C2",
        "score-prompts": "calculate all score–prompt cosine similarities",
        "embed-audio": "embed optional audio with CLaMP 3 SAAS",
        "analyze": "run preregistered work-level summaries, inference, and diagnostics",
        "make-run-manifest": "record run artifacts and model identities",
    }
    for name in COMMANDS:
        command = subparsers.add_parser(name, help=help_text[name], description=help_text[name])
        command.add_argument("--config", required=True, help="experiment YAML configuration")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_resolved_config(args.config)
        output = COMMANDS[args.command](config)
    except (
        AnalysisError,
        ConfigurationError,
        DataValidationError,
        EmbeddingError,
        FeatureError,
        FingerprintError,
        ManifestError,
        PromptError,
        SimilarityError,
    ) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
