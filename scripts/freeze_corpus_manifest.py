#!/usr/bin/env python3
"""Copy the current corpus manifest into a versioned study freeze directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from piano_clamp.data_loading import load_pipeline_config  # noqa: E402
from piano_clamp.study import StudyAuditError, freeze_study_manifest  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument("--label", required=True, help="Version label for the frozen manifest directory.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="Freeze even when the study readiness audit reports blocking errors.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_pipeline_config(args.config)
        destination = freeze_study_manifest(
            config,
            label=args.label,
            force=args.force,
            allow_errors=args.allow_errors,
        )
    except (StudyAuditError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
