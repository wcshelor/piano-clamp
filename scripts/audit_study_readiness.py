#!/usr/bin/env python3
"""Audit corpus registrations and passage/audio readiness without using a GPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from piano_clamp.data_loading import load_pipeline_config, read_corpus_manifest  # noqa: E402
from piano_clamp.study import build_study_readiness_report, write_study_readiness_report  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON report path. Defaults to analysis/reports/study_readiness.json under the resolved config.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_pipeline_config(args.config)
        rows = read_corpus_manifest(config["corpus_manifest"])
        report = build_study_readiness_report(
            rows,
            corpus_root=config["corpus_root"],
            alignment_root=config.get("alignment_root"),
            authorized_rights=tuple(config.get("authorized_rights_statuses", ())),
            source_manifest_path=config["corpus_manifest"],
        )
        output = args.output or (Path(config["analysis_root"]) / "reports" / "study_readiness.json")
        write_study_readiness_report(output, report)
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(output)
    return 1 if report["status"] == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
