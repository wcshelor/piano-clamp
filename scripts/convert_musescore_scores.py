#!/usr/bin/env python3
"""Convert native MuseScore scores into validated MusicXML ahead of embedding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from piano_clamp.musescore_conversion import (  # noqa: E402
    MuseScoreConversionError,
    convert_musescore_scores,
    summary_dict,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report-root", required=True, type=Path)
    parser.add_argument(
        "--musescore-bin",
        default="mscore",
        help="MuseScore CLI binary to call for --version and -o export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing exported .musicxml files instead of skipping them.",
    )
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = convert_musescore_scores(
            input_root=args.input_root,
            output_root=args.output_root,
            report_root=args.report_root,
            musescore_binary=args.musescore_bin,
            overwrite=args.overwrite,
        )
    except (MuseScoreConversionError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    payload = summary_dict(summary)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            f"Converted {summary.converted_count} of {summary.source_count} MuseScore scores "
            f"with {summary.failed_count} failures and {summary.skipped_count} skips."
        )
        print(f"MuseScore: {summary.musescore_version}")
        print(f"CSV report: {summary.csv_report}")
        print(f"JSON report: {summary.json_report}")
    return 1 if summary.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
