#!/usr/bin/env python3
"""Small end-to-end smoke test for symbolic or audio passage embedding."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from piano_clamp.clamp_backend import ClampBackend  # noqa: E402
from piano_clamp.data_loading import filter_manifest_rows, load_pipeline_config, read_corpus_manifest  # noqa: E402
from piano_clamp.embedding_io import sha256_path  # noqa: E402
from piano_clamp.passage_generation import (  # noqa: E402
    extract_aligned_audio_passage,
    generate_passage_records,
    load_alignment_times,
    materialize_symbolic_passage,
    parse_bar_ranges,
    rights_authorized,
)


def _method_spec(name: str) -> tuple[str, int | None]:
    mapping = {
        "complete": ("complete_movement", None),
        "4": ("fixed_bar_window", 4),
        "8": ("fixed_bar_window", 8),
        "16": ("fixed_bar_window", 16),
        "phrase": ("phrase", None),
        "cadence": ("cadence", None),
        "user": ("user_range", None),
    }
    if name not in mapping:
        raise ValueError(f"unsupported smoke-test passage mode: {name}")
    return mapping[name]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paths.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--modality", choices=("symbolic", "audio"), default="symbolic")
    parser.add_argument(
        "--source-material",
        choices=("auto", "score", "performance_midi", "audio"),
        default="auto",
    )
    parser.add_argument("--mode", choices=("complete", "4", "8", "16", "phrase", "cadence", "user"), default="16")
    parser.add_argument("--bar-range", action="append", help="START-END, required with --mode user")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--composer")
    parser.add_argument("--work-id")
    parser.add_argument("--recording-id")
    parser.add_argument(
        "--stride-bars",
        type=int,
        help="step size for fixed-bar windows; defaults to the window size, so smaller values create overlap",
    )
    parser.add_argument("--output", help="retain smoke outputs at this directory")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="only run the CLaMP runtime probe and write diagnostics without materializing passages",
    )
    args = parser.parse_args(argv)

    if args.limit < 1:
        parser.error("--limit must be positive")
    if args.stride_bars is not None and args.stride_bars < 1:
        parser.error("--stride-bars must be positive")

    try:
        source_material = args.source_material
        if source_material == "auto":
            source_material = "audio" if args.modality == "audio" else "score"
        if args.modality == "audio" and source_material != "audio":
            parser.error("--modality audio requires --source-material audio")
        if args.modality == "symbolic" and source_material not in {"score", "performance_midi"}:
            parser.error("--modality symbolic requires --source-material score or performance_midi")
        config = load_pipeline_config(args.config, device=args.device)
        manifest_rows = filter_manifest_rows(
            read_corpus_manifest(config["corpus_manifest"]),
            composer=args.composer,
            work_id=args.work_id,
        )
        if args.recording_id:
            manifest_rows = [row for row in manifest_rows if row["recording_id"] == args.recording_id]
        method, window = _method_spec(args.mode)
        user_ranges = parse_bar_ranges(args.bar_range or [])
        if method == "user_range" and not user_ranges:
            parser.error("--mode user requires at least one --bar-range START-END")
        passages = generate_passage_records(
            manifest_rows,
            corpus_root=config["corpus_root"],
            method=method,
            window_bars=window,
            stride_bars=args.stride_bars,
            user_ranges=user_ranges,
        )[: args.limit]
        if not passages:
            raise RuntimeError("no passages matched the smoke-test filters")

        backend = ClampBackend(config, model_space="c2" if args.modality == "symbolic" else "saas")
        retained = Path(args.output).resolve() if args.output else None
        context = tempfile.TemporaryDirectory(prefix="piano-clamp-passage-smoke-") if retained is None else None
        output = retained or Path(context.name)
        output.mkdir(parents=True, exist_ok=True)
        probe = backend.probe_runtime(
            workspace=output / "runtime_probe_work",
            log_path=output / "passage_smoke.log",
            modality="symbolic" if args.modality == "symbolic" else "audio",
        )
        (output / "runtime_probe.json").write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
        if args.preflight_only:
            report = {
                "status": "ok",
                "device": backend.device,
                "modality": args.modality,
                "source_material": source_material,
                "mode": args.mode,
                "preflight_only": True,
                "runtime_probe_path": str(output / "runtime_probe.json"),
                "output": str(output),
            }
            (output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(report)
            if context is not None:
                context.cleanup()
            return 0
        materialized = output / "materialized"

        staged_sources: dict[str, Path] = {}
        prepared_rows: list[dict[str, object]] = []
        events = (
            load_alignment_times(config["alignment_root"])
            if args.modality == "audio" or source_material == "performance_midi"
            else {}
        )
        allowed_rights = list(config.get("authorized_rights_statuses", []))
        for record in passages:
            entry = dict(record)
            try:
                if args.modality == "symbolic":
                    path = materialize_symbolic_passage(
                        entry,
                        corpus_root=config["corpus_root"],
                        destination=materialized,
                        source_material=source_material,
                        events=events,
                    )
                else:
                    if not rights_authorized(entry.get("rights_status", ""), allowed_rights):
                        raise RuntimeError("audio rights_status is not explicitly authorized")
                    path, onset, offset = extract_aligned_audio_passage(
                        entry,
                        corpus_root=config["corpus_root"],
                        events=events,
                        destination=materialized,
                        sample_rate=int(config.get("audio_sample_rate", 24000)),
                    )
                    entry["onset_seconds"] = onset
                    entry["offset_seconds"] = offset
                entry["content_hash"] = sha256_path(path)
                staged_sources[str(entry["passage_id"])] = path
                entry["status"] = "prepared"
                entry["error_message"] = ""
            except Exception as exc:
                entry["status"] = "failed"
                entry["error_message"] = str(exc)
            prepared_rows.append(entry)

        ok_sources = {
            passage_id: path
            for passage_id, path in staged_sources.items()
            if any(row["passage_id"] == passage_id and row["status"] == "prepared" for row in prepared_rows)
        }
        if not ok_sources:
            raise RuntimeError("no passages could be materialized for smoke testing")

        log_path = output / "passage_smoke.log"
        if args.modality == "symbolic":
            vectors, errors = backend.embed_symbolic(
                ok_sources,
                workspace=output / "backend_work",
                log_path=log_path,
            )
        else:
            vectors, errors = backend.embed_audio_files(
                ok_sources,
                workspace=output / "backend_work",
                log_path=log_path,
            )

        success_ids = []
        for row in prepared_rows:
            passage_id = str(row["passage_id"])
            if passage_id in vectors:
                vector = np.asarray(vectors[passage_id], dtype=np.float32)
                if vector.shape != (768,) or not np.isfinite(vector).all():
                    row["status"] = "failed"
                    row["error_message"] = f"invalid vector shape or values: {vector.shape}"
                else:
                    row["status"] = "success"
                    row["error_message"] = ""
                    success_ids.append(passage_id)
                    np.save(output / f"{passage_id}.npy", vector[None, :], allow_pickle=False)
            elif passage_id in errors:
                row["status"] = "failed"
                row["error_message"] = errors[passage_id]

        report = {
            "status": "ok" if success_ids else "failed",
            "device": backend.device,
            "modality": args.modality,
            "source_material": source_material,
            "mode": args.mode,
            "stride_bars": args.stride_bars,
            "requested_limit": args.limit,
            "runtime_probe_path": str(output / "runtime_probe.json"),
            "prepared_count": sum(row["status"] in {"prepared", "success"} for row in prepared_rows),
            "success_count": len(success_ids),
            "failure_count": sum(row["status"] == "failed" for row in prepared_rows),
            "success_ids": success_ids,
            "output": str(output),
            "rows": prepared_rows,
        }
        (output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(report)
        if context is not None and success_ids:
            context.cleanup()
        return 0 if success_ids else 1
    except Exception as exc:
        parser.exit(2, f"passage smoke test failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
