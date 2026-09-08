#!/usr/bin/env python3
"""Validate every matrix, metadata table, source record, and run manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.data_loading import load_pipeline_config  # noqa: E402
from piano_clamp.validation import validate_embedding_outputs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    args = parser.parse_args(argv)
    try:
        report = validate_embedding_outputs(load_pipeline_config(args.config))
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")
    print(report["json_report"])
    print(report["text_report"])
    return 1 if report["status"] != "ok" else 0


if __name__ == "__main__":
    raise SystemExit(main())

