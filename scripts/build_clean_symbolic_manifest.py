#!/usr/bin/env python3
"""Write a filtered CPC manifest that excludes ambiguous symbolic movements."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from piano_clamp.data_loading import load_pipeline_config, read_corpus_manifest  # noqa: E402
from piano_clamp.study import StudyAuditError, write_clean_symbolic_manifest_bundle  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis") / "filtered_manifests" / "clean_symbolic",
        help="Directory that will receive the filtered manifest and JSON report.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_pipeline_config(args.config)
        rows = read_corpus_manifest(config["corpus_manifest"])
        paths = write_clean_symbolic_manifest_bundle(
            args.output_root,
            rows=rows,
            source_manifest_path=config["corpus_manifest"],
            force=args.force,
        )
    except (StudyAuditError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(paths["manifest_path"])
    print(paths["report_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
