#!/usr/bin/env python3
"""Snapshot any existing live embedding bundles into the append-only store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.data_loading import load_pipeline_config  # noqa: E402
from piano_clamp.embedding_store import EmbeddingStoreError, snapshot_bundle  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_pipeline_config(args.config)
        root = Path(config["output_root"])
        locations = (
            root / "text",
            root / "text_saas",
            root / "symbolic",
            root / "audio",
            root / "audio" / "chunks",
            root / "passages",
            root / "passages" / "audio",
        )
        imported = 0
        for directory in locations:
            metadata = directory / "metadata.json"
            if not metadata.is_file():
                continue
            snapshot_bundle(
                directory,
                store_root=config["embedding_store_root"],
                output_root=config["output_root"],
            )
            imported += 1
    except (EmbeddingStoreError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(imported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
