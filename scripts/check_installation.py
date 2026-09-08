#!/usr/bin/env python3
"""Inspect CLaMP wiring, checkpoints, corpus assets, dependencies, and device."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.clamp_backend import DISCOVERED_API, inspect_backend  # noqa: E402
from piano_clamp.data_loading import (  # noqa: E402
    PipelineConfigurationError,
    load_pipeline_config,
    read_corpus_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--json", action="store_true", help="emit one JSON document")
    args = parser.parse_args(argv)
    report: dict[str, object] = {"api": DISCOVERED_API, "errors": [], "warnings": []}
    try:
        config = load_pipeline_config(args.config, device=args.device)
    except PipelineConfigurationError as exc:
        report["errors"] = [str(exc)]
        # Static checks remain useful before a corpus path has been configured.
        report["c2_checkpoint_available"] = any((ROOT / "models" / "clamp3-c2").glob("*.pth"))
        report["vendor_available"] = (ROOT / "vendor" / "clamp3" / "clamp3_embd.py").is_file()
    else:
        backend = inspect_backend(config)
        report.update(backend)
        required_runtime_modules = {
            "music21": importlib.util.find_spec("music21") is not None,
            "soundfile": importlib.util.find_spec("soundfile") is not None,
        }
        report["required_runtime_modules"] = required_runtime_modules
        missing_runtime_modules = [
            name for name, available in required_runtime_modules.items() if not available
        ]
        if missing_runtime_modules:
            report["errors"].append(
                "required runtime dependencies missing: " + ", ".join(missing_runtime_modules)
            )
        for key in ("missing_text_modules", "missing_symbolic_modules"):
            missing = backend[key]
            if missing:
                report["errors"].append(f"{key}: {', '.join(missing)}")
        if backend["missing_audio_modules"]:
            report["warnings"].append(
                f"optional audio dependencies missing: {', '.join(backend['missing_audio_modules'])}"
            )
        rows = read_corpus_manifest(config["corpus_manifest"])
        root = Path(config["corpus_root"])
        report["corpus"] = {
            "manifest_rows": len(rows),
            "composer_rows": dict(Counter(row["composer"] for row in rows)),
            "unique_movements": len({row["movement_id"] for row in rows}),
            "unique_recordings": len({row["recording_id"] for row in rows if row["recording_id"]}),
            "local_supported_scores": sum(
                bool(value)
                and Path(value).suffix.lower() in {".mxl", ".musicxml", ".xml", ".mid", ".midi"}
                and (root / value).is_file()
                for row in rows
                for value in [row["mxl_path"] or row["score_path"] or row["midi_path"]]
            ),
            "local_audio_rows": sum(bool(row["audio_path"]) and (root / row["audio_path"]).is_file() for row in rows),
        }
        if not backend["saas_checkpoint_available"]:
            report["warnings"].append("SAAS checkpoint absent: audio embedding is unavailable")
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
