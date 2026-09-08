#!/usr/bin/env python3
"""Append render-backed rows into a new local CPC adapter bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from piano_clamp.cpc_adapter import (  # noqa: E402
    CpcAdapterError,
    supplement_cpc_adapter_with_render_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", required=True, type=Path)
    parser.add_argument("--adapter-root", required=True, type=Path)
    parser.add_argument("--render-manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--export-manifest",
        type=Path,
        default=Path("manifests/exports/piano-clamp-chopin-mozart.csv"),
        help="CPC 1.1 export path, relative to --corpus-root unless absolute.",
    )
    parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="Validate score/audio digests from metadata without re-reading every file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = supplement_cpc_adapter_with_render_manifest(
            corpus_root=args.corpus_root,
            adapter_root=args.adapter_root,
            render_manifest=args.render_manifest,
            output_root=args.output_root,
            export_manifest=args.export_manifest,
            verify_hashes=not args.skip_hash_verification,
        )
    except (CpcAdapterError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(
        f"Added {summary.added_rows} recordings and {summary.added_timing_rows} measure timings; "
        f"new adapter has {summary.total_rows} recordings."
    )
    print(f"Adapter: {summary.output_root}")
    print(f"Render manifest: {summary.render_manifest}")
    print(f"Unmatched render rows: {summary.unmatched_render_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
