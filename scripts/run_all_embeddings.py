#!/usr/bin/env python3
"""Run selected embedding and analysis stages with progress logging."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.power import add_caffeinate_arguments, maybe_caffeinate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--symbolic", action="store_true")
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--passages", action="store_true")
    parser.add_argument(
        "--passage-source-material",
        action="append",
        choices=("score", "performance_midi", "audio"),
        help="repeat to select passage material families; defaults to all when --passages is set",
    )
    parser.add_argument("--similarities", action="store_true")
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    add_caffeinate_arguments(parser)
    args = parser.parse_args(argv)
    selected = any(
        (args.text, args.symbolic, args.audio, args.passages, args.similarities, args.index, args.validate)
    )
    if not selected:
        parser.error("enable at least one stage")
    stages = []
    passage_materials = args.passage_source_material or ["score", "performance_midi", "audio"]
    for enabled, script in (
        (args.text, "embed_text_prompts.py"),
        (args.symbolic, "embed_symbolic_scores.py"),
        (args.audio, "embed_audio.py"),
    ):
        if enabled:
            command = [sys.executable, str(ROOT / "scripts" / script), "--config", args.config]
            if args.device and script.startswith("embed_"):
                command.extend(["--device", args.device])
            if args.force and script not in {"validate_embeddings.py"}:
                command.append("--force")
            if args.caffeinate and script in {
                "embed_text_prompts.py",
                "embed_symbolic_scores.py",
                "embed_audio.py",
            }:
                command.extend(["--caffeinate", "--caffeinate-flags", args.caffeinate_flags])
            stages.append((script, command))
    if args.passages:
        for source_material in passage_materials:
            command = [
                sys.executable,
                str(ROOT / "scripts" / "embed_passages.py"),
                "--config",
                args.config,
                "--source-material",
                source_material,
                "--modality",
                "audio" if source_material == "audio" else "symbolic",
            ]
            if args.device:
                command.extend(["--device", args.device])
            if args.force:
                command.append("--force")
            if args.caffeinate:
                command.extend(["--caffeinate", "--caffeinate-flags", args.caffeinate_flags])
            stages.append((f"embed_passages.py:{source_material}", command))
    for enabled, script in (
        (args.index, "build_embedding_index.py"),
        (args.similarities, "compute_similarities.py"),
        (args.validate, "validate_embeddings.py"),
    ):
        if enabled:
            command = [sys.executable, str(ROOT / "scripts" / script), "--config", args.config]
            if args.force and script not in {"validate_embeddings.py"}:
                command.append("--force")
            stages.append((script, command))
    failures = []
    with maybe_caffeinate(args.caffeinate, flags=args.caffeinate_flags):
        for name, command in stages:
            timestamp = datetime.now(timezone.utc).isoformat()
            print(f"[{timestamp}] starting {name}: {shlex.join(command)}", flush=True)
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                failures.append({"stage": name, "exit_code": result.returncode})
                if not args.continue_on_error:
                    break
    if failures:
        print({"status": "failed", "failures": failures})
        return 1
    print({"status": "ok", "stages": [name for name, _ in stages]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
