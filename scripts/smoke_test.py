#!/usr/bin/env python3
"""Real CPU-capable CLaMP smoke test on three prompts and one local score."""

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
from piano_clamp.data_loading import (  # noqa: E402
    load_pipeline_config,
    load_prompt_bank,
    read_corpus_manifest,
    select_symbolic_records,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/embedding_config.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--output", help="retain smoke outputs at this directory")
    args = parser.parse_args(argv)
    try:
        config = load_pipeline_config(args.config, device=args.device)
        prompts = load_prompt_bank(config["prompt_bank"])[:3]
        scores = select_symbolic_records(
            read_corpus_manifest(config["corpus_manifest"]), config["corpus_root"]
        )
        score = next((row for row in scores if row["status"] == "pending"), None)
        backend = ClampBackend(config, model_space="c2")
        retained = Path(args.output).resolve() if args.output else None
        context = tempfile.TemporaryDirectory(prefix="piano-clamp-smoke-") if retained is None else None
        output = retained or Path(context.name)
        output.mkdir(parents=True, exist_ok=True)
        probe = backend.probe_runtime(
            workspace=output / "runtime_probe_work",
            log_path=output / "smoke.log",
            modality="symbolic",
        )
        (output / "runtime_probe.json").write_text(json.dumps(probe, indent=2) + "\n", encoding="utf-8")
        text_vectors, text_errors = backend.embed_texts(
            {row["prompt_id"]: row["prompt_text"] for row in prompts},
            workspace=output / "text_work",
            log_path=output / "smoke.log",
        )
        if text_errors or len(text_vectors) != 3:
            raise RuntimeError(f"text smoke failed: {text_errors}")
        text_matrix = np.stack([text_vectors[row["prompt_id"]] for row in prompts])
        if text_matrix.shape != (3, 768) or not np.isfinite(text_matrix).all():
            raise RuntimeError(f"invalid text smoke matrix: {text_matrix.shape}")
        np.save(output / "text_embeddings.npy", text_matrix, allow_pickle=False)
        score_count = 0
        if score is not None:
            score_vectors, score_errors = backend.embed_symbolic(
                {score["score_id"]: score["resolved_source_path"]},
                workspace=output / "score_work",
                log_path=output / "smoke.log",
            )
            if score_errors or score["score_id"] not in score_vectors:
                raise RuntimeError(f"symbolic smoke failed: {score_errors}")
            score_matrix = score_vectors[score["score_id"]][None, :]
            if score_matrix.shape != (1, 768) or not np.isfinite(score_matrix).all():
                raise RuntimeError(f"invalid symbolic smoke matrix: {score_matrix.shape}")
            np.save(output / "symbolic_embedding.npy", score_matrix, allow_pickle=False)
            score_count = 1
        report = {
            "status": "ok",
            "device": backend.device,
            "text_prompts": 3,
            "symbolic_scores": score_count,
            "embedding_dimension": 768,
            "runtime_probe_path": str(output / "runtime_probe.json"),
            "output": str(output),
        }
        (output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(report)
        if context is not None:
            context.cleanup()
        return 0
    except Exception as exc:
        parser.exit(2, f"smoke test failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
