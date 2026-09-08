#!/usr/bin/env python3
"""Export cached prepared passage windows for manual sheet-music review."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.data_loading import filter_manifest_rows, load_pipeline_config, read_corpus_manifest  # noqa: E402
from piano_clamp.passage_artifacts import export_review_passages  # noqa: E402
from piano_clamp.passage_generation import generate_passage_records, parse_bar_ranges  # noqa: E402


def _method_spec(name: str) -> tuple[str, int | None]:
    mapping = {
        "complete_movement": ("complete_movement", None),
        "complete": ("complete_movement", None),
        "fixed_4_bars": ("fixed_bar_window", 4),
        "4": ("fixed_bar_window", 4),
        "fixed_8_bars": ("fixed_bar_window", 8),
        "8": ("fixed_bar_window", 8),
        "fixed_16_bars": ("fixed_bar_window", 16),
        "16": ("fixed_bar_window", 16),
        "phrase": ("phrase", None),
        "cadence": ("cadence", None),
        "user_range": ("user_range", None),
        "user": ("user_range", None),
    }
    if name not in mapping:
        raise ValueError(f"unknown passage method: {name}")
    return mapping[name]


def _resume_directory(output_root: Path, source_material: str) -> Path:
    output = output_root / "passages"
    if source_material != "score":
        output = output / source_material
    return output / ".resume"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument("--composer")
    parser.add_argument("--work-id")
    parser.add_argument("--recording-id")
    parser.add_argument(
        "--mode",
        action="append",
        choices=("complete", "4", "8", "16", "phrase", "cadence", "user"),
        help="repeat to combine passage-generation modes; defaults to the config list",
    )
    parser.add_argument("--bar-range", action="append", help="START-END for user mode")
    parser.add_argument("--stride-bars", type=int)
    parser.add_argument("--overlap-half", action="store_true")
    parser.add_argument("--passage-id", action="append")
    parser.add_argument("--status", choices=("all", "success", "failed", "pending"), default="all")
    parser.add_argument(
        "--source-material",
        choices=("score", "performance_midi", "audio"),
        default="score",
        help="passage material cache to export",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis") / "passage_review",
        help="directory that will receive copied prepared passages and a review manifest",
    )
    parser.add_argument(
        "--musescore-bin",
        default="mscore",
        help="MuseScore CLI binary to call for PNG preview export.",
    )
    parser.add_argument(
        "--preview-renderer",
        choices=("auto", "musescore", "pianoroll"),
        default="auto",
        help="preview backend: try MuseScore first, require MuseScore, or force the built-in piano-roll renderer",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="skip PNG preview rendering and export only the prepared score files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.stride_bars is not None and args.stride_bars < 1:
        raise SystemExit("--stride-bars must be positive")
    if args.overlap_half and args.stride_bars is not None:
        raise SystemExit("--overlap-half cannot be combined with --stride-bars")
    try:
        config = load_pipeline_config(args.config)
        manifest_rows = filter_manifest_rows(
            read_corpus_manifest(config["corpus_manifest"]),
            composer=args.composer,
            work_id=args.work_id,
        )
        if args.recording_id:
            manifest_rows = [row for row in manifest_rows if row["recording_id"] == args.recording_id]
        modes = args.mode or list(config.get("passage_methods", ["complete_movement"]))
        user_ranges = parse_bar_ranges(args.bar_range or [])
        records = []
        seen = set()
        counts: Counter[str] = Counter()
        for mode in modes:
            method, window = _method_spec(mode)
            if method == "user_range" and not user_ranges:
                raise ValueError("user_range mode requires at least one --bar-range START-END")
            effective_stride = args.stride_bars
            if method == "fixed_bar_window" and window is not None and args.overlap_half and effective_stride is None:
                effective_stride = max(1, window // 2)
            generated = generate_passage_records(
                manifest_rows,
                corpus_root=config["corpus_root"],
                method=method,
                window_bars=window,
                stride_bars=effective_stride,
                user_ranges=user_ranges,
            )
            for row in generated:
                if row["passage_id"] not in seen:
                    seen.add(row["passage_id"])
                    records.append(row)
                    counts[str(row.get("passage_generation_method", mode))] += 1
        records.sort(key=lambda row: row["passage_id"])

        resume_dir = _resume_directory(Path(config["output_root"]), args.source_material)
        state_path = resume_dir / "state.json"
        if not state_path.is_file():
            raise ValueError(f"resume state does not exist: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        status_map = {
            str(key): {str(k): str(v) for k, v in value.items()}
            for key, value in dict(state.get("record_status", {})).items()
            if isinstance(value, dict)
        }
        label_parts = [args.composer or "all", "passages", args.source_material, args.status]
        destination = args.output_dir / "-".join(part.replace(" ", "_").lower() for part in label_parts)
        result = export_review_passages(
            destination,
            records=records,
            status_map=status_map,
            prepared_root=resume_dir / "prepared",
            status_filter=args.status,
            passage_ids=args.passage_id or (),
            limit=args.limit,
            render_png=not args.no_png,
            musescore_binary=args.musescore_bin,
            preview_renderer=args.preview_renderer,
        )
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(result["manifest_path"])
    print(result["prepared_directory"])
    print(result["png_directory"])
    print(result["exported_rows"])
    print(result["copied_files"])
    print(result["png_files"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
