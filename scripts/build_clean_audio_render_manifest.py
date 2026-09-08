#!/usr/bin/env python3
"""Write a filtered CPC manifest containing only render rows reconciled to the active export."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from piano_clamp.data_loading import load_pipeline_config, read_corpus_manifest  # noqa: E402
from piano_clamp.study import StudyAuditError, write_clean_audio_render_manifest_bundle  # noqa: E402


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument(
        "--render-manifest",
        type=Path,
        required=True,
        help="CSV emitted by the corpus-side rendering step.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("analysis") / "filtered_manifests" / "clean_audio_renders",
        help="Directory that will receive the reconciled manifest and JSON report.",
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_pipeline_config(args.config)
        export_rows = read_corpus_manifest(config["corpus_manifest"])
        render_rows = _read_csv_rows(args.render_manifest)
        paths = write_clean_audio_render_manifest_bundle(
            args.output_root,
            render_rows=render_rows,
            export_rows=export_rows,
            render_manifest_path=args.render_manifest,
            export_manifest_path=config["corpus_manifest"],
            force=args.force,
        )
    except (StudyAuditError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(paths["manifest_path"])
    print(paths["report_path"])
    print(paths["unmatched_render_rows_path"])
    print(paths["stale_export_rows_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
