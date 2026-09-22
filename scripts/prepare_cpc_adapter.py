#!/usr/bin/env python3
"""Build Piano CLaMP's local adapter from canonical CPC tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from piano_clamp.cpc_adapter import CpcAdapterError, build_cpc_adapter  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--export-manifest",
        type=Path,
        default=None,
        help=(
            "Optional deprecated CPC export path for compatibility. If omitted, "
            "the adapter derives its study subset directly from canonical CPC tables."
        ),
    )
    parser.add_argument(
        "--composer",
        action="append",
        dest="composers",
        help=(
            "Composer to retain; repeat as needed. When omitted, all eligible "
            "composers are retained. Filtering is an explicit opt-in."
        ),
    )
    parser.add_argument(
        "--all-composers",
        action="store_true",
        help=(
            "Retain every eligible composer in the canonical CPC tables. "
            "This is now the default when --composer is omitted and is kept for "
            "backwards compatibility with legacy Chopin/Mozart examples."
        ),
    )
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="Validate canonical hashes without re-reading every score/audio file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.all_composers and args.composers:
        raise SystemExit("--all-composers cannot be combined with --composer")
    try:
        # Default is all eligible composers; filtering requires explicit --composer values.
        composers = None
        if args.composers:
            composers = args.composers
        elif args.all_composers:
            composers = None
        summary = build_cpc_adapter(
            corpus_root=args.corpus_root,
            output_root=args.output_root,
            export_manifest=args.export_manifest,
            composers=composers,
            verify_hashes=not args.skip_hash_verification,
        )
    except (CpcAdapterError, OSError, ValueError) as exc:
        import traceback

        traceback.print_exc()
        raise SystemExit(f"error: {exc}\n{traceback.format_exc()}") from exc
    print(
        f"Prepared {summary.imported_rows} recordings and {summary.timing_rows} measure timings "
        f"from {summary.corpus_release}; excluded {summary.excluded_rows} rows."
    )
    print(f"Manifest: {summary.score_manifest}")
    print(f"Alignments: {summary.alignment_events}")
    print(f"Readiness: {summary.readiness_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
