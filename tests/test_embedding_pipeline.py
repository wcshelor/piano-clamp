import csv
import json
import stat
import tempfile
import unittest
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from piano_clamp.data_loading import load_pipeline_config, load_prompt_bank
from piano_clamp.embedding_io import (
    EmbeddingIOError,
    align_success_rows,
    read_embedding_bundle,
    write_embedding_bundle,
)
from piano_clamp.embedding_store import snapshot_bundle
from piano_clamp.passage_artifacts import export_review_passages, write_failed_passage_rows
from piano_clamp.passage_generation import stable_passage_id
from tests.test_features import SYNTHETIC_MUSICXML


class PromptBankTests(unittest.TestCase):
    def test_prompt_bank_minimums_and_pairs(self):
        root = Path(__file__).resolve().parents[1]
        rows = load_prompt_bank(root / "prompts" / "prompt_bank_v1.csv")
        counts = Counter(row["family"] for row in rows)
        self.assertGreaterEqual(counts["literal"], 20)
        self.assertGreaterEqual(counts["texture"], 15)
        self.assertGreaterEqual(counts["emotion"], 15)
        self.assertGreaterEqual(counts["performance"], 10)
        self.assertGreaterEqual(counts["historical"] + counts["comparative"], 10)
        self.assertGreaterEqual(counts["matched_pair"], 20)
        poles = {}
        for row in rows:
            if row["family"] == "matched_pair":
                poles.setdefault(row["subfamily"], set()).add(row["polarity"])
        self.assertTrue(poles)
        self.assertTrue(all(value == {"positive", "negative"} for value in poles.values()))

    def test_config_copy_resolves_to_canonical_bank(self):
        root = Path(__file__).resolve().parents[1]
        canonical = (root / "prompts" / "prompt_bank_v1.csv").read_bytes()
        configured = (root / "configs" / "prompt_bank_v1.csv").read_bytes()
        self.assertEqual(configured, canonical)

    def test_embedding_config_automatically_overlays_local_paths_yaml(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = root / "configs"
            configs.mkdir()
            (configs / "embedding_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "corpus_root": "${PIANO_CLAMP_CORPUS_ROOT}",
                        "adapter_root": "data/cpc_adapter/example",
                        "corpus_manifest": "piano_clamp_manifest.csv",
                        "score_root": "data/processed/scores",
                        "audio_root": "data/processed/performances",
                        "alignment_root": ".",
                        "output_root": "embeddings",
                        "analysis_root": "analysis",
                        "log_root": "logs",
                        "temporary_root": "tmp",
                        "vendor_root": "vendor/clamp3",
                        "checkpoint_path": "models/c2.pth",
                        "saas_checkpoint_path": "models/saas.pth",
                        "prompt_bank": "prompts/prompt_bank_v1.csv",
                    }
                ),
                encoding="utf-8",
            )
            corpus = root / "corpus"
            adapter = root / "adapter"
            (configs / "paths.yaml").write_text(
                yaml.safe_dump(
                    {
                        "corpus_root": str(corpus),
                        "adapter_root": str(adapter),
                        "output_root": str(root / "run_outputs"),
                    }
                ),
                encoding="utf-8",
            )
            resolved = load_pipeline_config(
                configs / "embedding_config.yaml",
                environ={},
                root=root,
            )
            self.assertEqual(Path(resolved["corpus_root"]), corpus.resolve())
            self.assertEqual(Path(resolved["adapter_root"]), adapter.resolve())
            self.assertEqual(Path(resolved["output_root"]), (root / "run_outputs").resolve())

    def test_embedding_config_can_disable_local_paths_overlay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = root / "configs"
            configs.mkdir()
            (configs / "embedding_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "corpus_root": "${PIANO_CLAMP_CORPUS_ROOT}",
                        "use_local_paths_overlay": False,
                        "adapter_root": "data/cpc_symbolic_adapter/bach-v0",
                        "corpus_manifest": "piano_clamp_manifest.csv",
                        "score_root": "data/processed/scores",
                        "audio_root": "data/processed/performances",
                        "alignment_root": ".",
                        "output_root": "bach_outputs",
                        "analysis_root": "analysis",
                        "log_root": "logs",
                        "temporary_root": "tmp",
                        "vendor_root": "vendor/clamp3",
                        "checkpoint_path": "models/c2.pth",
                        "saas_checkpoint_path": "models/saas.pth",
                        "prompt_bank": "prompts/prompt_bank_v1.csv",
                    }
                ),
                encoding="utf-8",
            )
            corpus = root / "corpus"
            (configs / "paths.yaml").write_text(
                yaml.safe_dump(
                    {
                        "corpus_root": str(corpus / "override"),
                        "adapter_root": str(root / "override_adapter"),
                        "output_root": str(root / "override_outputs"),
                    }
                ),
                encoding="utf-8",
            )
            resolved = load_pipeline_config(
                configs / "embedding_config.yaml",
                environ={"PIANO_CLAMP_CORPUS_ROOT": str(corpus)},
                root=root,
            )
            self.assertEqual(Path(resolved["corpus_root"]), corpus.resolve())
            self.assertEqual(
                Path(resolved["adapter_root"]),
                (root / "data/cpc_symbolic_adapter/bach-v0").resolve(),
            )
            self.assertEqual(Path(resolved["output_root"]), (root / "bach_outputs").resolve())


class EmbeddingBundleTests(unittest.TestCase):
    def test_table_rows_align_with_matrix_and_guard_overwrite(self):
        records = [
            {"prompt_id": "p1", "status": "pending", "error_message": ""},
            {"prompt_id": "p2", "status": "pending", "error_message": ""},
        ]
        vectors = {"p1": np.ones(4), "p2": np.array([1.0, 2.0, 3.0, 4.0])}
        matrix, rows = align_success_rows(
            records, vectors, id_field="prompt_id", expected_dimension=4
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_embedding_bundle(
                root,
                matrix,
                rows,
                matrix_filename="prompt_embeddings.npy",
                table_filename="prompt_embeddings.csv",
                fields=("prompt_id", "embedding_row", "embedding_dimension", "status", "error_message"),
                id_field="prompt_id",
                metadata={"model": "test", "model_space": "test", "checkpoint_path": "test", "checkpoint_hash": "test"},
            )
            loaded, loaded_rows, metadata = read_embedding_bundle(root)
            self.assertEqual(loaded.shape, (2, 4))
            self.assertEqual(len(loaded_rows), loaded.shape[0])
            self.assertEqual(metadata["embedding_shape"], [2, 4])
            with self.assertRaisesRegex(EmbeddingIOError, "refusing to overwrite"):
                write_embedding_bundle(
                    root,
                    matrix,
                    rows,
                    matrix_filename="prompt_embeddings.npy",
                    table_filename="prompt_embeddings.csv",
                    fields=("prompt_id", "embedding_row", "embedding_dimension", "status", "error_message"),
                    id_field="prompt_id",
                    metadata={},
                )

    def test_explicit_failure_does_not_claim_a_matrix_row(self):
        records = [
            {"score_id": "ok", "status": "pending", "error_message": ""},
            {"score_id": "bad", "status": "unsupported", "error_message": "MSCX"},
        ]
        matrix, rows = align_success_rows(
            records, {"ok": np.ones(3)}, id_field="score_id", expected_dimension=3
        )
        self.assertEqual(matrix.shape, (1, 3))
        self.assertEqual(rows[0]["embedding_row"], 0)
        self.assertEqual(rows[1]["embedding_row"], "")
        self.assertEqual(rows[1]["status"], "unsupported")

    def test_embedding_store_snapshots_bundle_and_skips_duplicate_identities(self):
        rows = [
            {
                "passage_id": "p4",
                "composer": "Frédéric Chopin",
                "period": "Romantic",
                "composition_id": "cmp1",
                "work_id": "w1",
                "movement_id": "m1",
                "recording_id": "r1",
                "performer": "Performer",
                "bar_start": 1,
                "bar_end": 4,
                "score_path": "scores/p4.musicxml",
                "audio_path": "",
                "midi_path": "",
                "alignment_path": "",
                "annotation_path": "",
                "rights_status": "licensed_for_research",
                "audio_origin": "source_recording",
                "passage_generation_method": "fixed_4_bars",
                "embedding_modality": "symbolic",
                "onset_seconds": "",
                "offset_seconds": "",
                "content_hash": "hash-p4",
                "embedding_row": 0,
                "embedding_dimension": 4,
                "status": "success",
                "error_message": "",
            },
            {
                "passage_id": "p8",
                "composer": "Frédéric Chopin",
                "period": "Romantic",
                "composition_id": "cmp1",
                "work_id": "w1",
                "movement_id": "m1",
                "recording_id": "r1",
                "performer": "Performer",
                "bar_start": 1,
                "bar_end": 8,
                "score_path": "scores/p8.musicxml",
                "audio_path": "",
                "midi_path": "",
                "alignment_path": "",
                "annotation_path": "",
                "rights_status": "licensed_for_research",
                "audio_origin": "source_recording",
                "passage_generation_method": "fixed_8_bars",
                "embedding_modality": "symbolic",
                "onset_seconds": "",
                "offset_seconds": "",
                "content_hash": "hash-p8",
                "embedding_row": 1,
                "embedding_dimension": 4,
                "status": "success",
                "error_message": "",
            },
        ]
        metadata = {
            "run_id": "20260822T000000Z-test",
            "timestamp": "2026-08-22T00:00:00+00:00",
            "stage": "passages_symbolic",
            "model": "CLaMP 3",
            "model_space": "c2",
            "checkpoint_path": "checkpoint.pth",
            "checkpoint_hash": "chk",
            "configuration_hash": "cfg",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "embeddings"
            store = root / "embedding_store"
            write_embedding_bundle(
                output / "passages",
                np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float),
                rows,
                matrix_filename="passage_embeddings.npy",
                table_filename="passage_embeddings.csv",
                fields=tuple(rows[0]),
                id_field="passage_id",
                metadata=metadata,
            )
            first = snapshot_bundle(output / "passages", store_root=store, output_root=output)
            second = snapshot_bundle(output / "passages", store_root=store, output_root=output)
            self.assertEqual(first["appended"], 2)
            self.assertEqual(second["appended"], 0)
            self.assertEqual(second["duplicates"], 2)
            with (store / "registry.csv").open(newline="", encoding="utf-8") as handle:
                registry_rows = list(csv.DictReader(handle))
            self.assertEqual(len(registry_rows), 2)
            self.assertEqual({row["window_bars"] for row in registry_rows}, {"4", "8"})
            self.assertEqual({row["item_type"] for row in registry_rows}, {"passage_symbolic"})
            self.assertTrue((store / "runs" / metadata["run_id"] / "passages" / "metadata.json").is_file())

    def test_failed_passage_report_and_review_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resume = root / ".resume"
            prepared = resume / "prepared"
            prepared.mkdir(parents=True)
            window = prepared / "p1.musicxml"
            window.write_text("<score-partwise/>", encoding="utf-8")
            records = [
                {
                    "passage_id": "p1",
                    "composer": "Frédéric Chopin",
                    "composition_id": "cmp1",
                    "work_id": "w1",
                    "movement_id": "m1",
                    "recording_id": "r1",
                    "bar_start": 1,
                    "bar_end": 4,
                    "passage_generation_method": "fixed_4_bars",
                    "embedding_modality": "symbolic",
                    "status": "pending",
                    "error_message": "",
                    "content_hash": "",
                },
                {
                    "passage_id": "p2",
                    "composer": "Frédéric Chopin",
                    "composition_id": "cmp1",
                    "work_id": "w1",
                    "movement_id": "m1",
                    "recording_id": "r1",
                    "bar_start": 5,
                    "bar_end": 8,
                    "passage_generation_method": "fixed_4_bars",
                    "embedding_modality": "symbolic",
                    "status": "pending",
                    "error_message": "",
                    "content_hash": "",
                },
            ]
            status_map = {
                "p1": {
                    "status": "failed",
                    "error_message": "symbolic preprocessing failed",
                    "content_hash": "h1",
                    "prepared_path": "prepared/p1.musicxml",
                },
                "p2": {
                    "status": "pending",
                    "error_message": "",
                    "content_hash": "h2",
                    "prepared_path": "",
                },
            }
            failed_path = write_failed_passage_rows(root / "failed_passages.csv", records=records, status_map=status_map)
            with failed_path.open(newline="", encoding="utf-8") as handle:
                failed_rows = list(csv.DictReader(handle))
            self.assertEqual(len(failed_rows), 1)
            self.assertEqual(failed_rows[0]["passage_id"], "p1")
            review = export_review_passages(
                root / "review",
                records=records,
                status_map=status_map,
                prepared_root=prepared,
                status_filter="failed",
                render_png=False,
            )
            self.assertEqual(review["exported_rows"], 1)
            self.assertTrue((root / "review" / "prepared" / "p1.musicxml").is_file())


class PassageIdTests(unittest.TestCase):
    def test_stable_id_includes_recording_and_collection(self):
        record = {
            "composer": "Frédéric Chopin",
            "work_id": "chopin-op006-no01",
            "movement_id": "chopin-op006-no01-m01",
            "recording_id": "performer:take-1",
            "source_collection": "example",
        }
        first = stable_passage_id(record, 1, 8)
        second = stable_passage_id(record, 1, 8)
        self.assertEqual(first, second)
        self.assertIn("performer-take-1", first)
        self.assertTrue(first.endswith("0001_0008"))


class PassageReviewExportTests(unittest.TestCase):
    def _write_fake_musescore(self, root: Path) -> Path:
        binary = root / "fake_mscore"
        binary.write_text(
            """#!/usr/bin/env python3
import sys
from pathlib import Path

if len(sys.argv) == 2 and sys.argv[1] == "--version":
    print("MuseScore 4.5.2")
    raise SystemExit(0)
if len(sys.argv) != 4 or sys.argv[1] != "-o":
    print("unexpected arguments", file=sys.stderr)
    raise SystemExit(2)
output = Path(sys.argv[2])
source = Path(sys.argv[3])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(f"preview for {source.name}\\n", encoding="utf-8")
""",
            encoding="utf-8",
        )
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        return binary

    def test_review_export_renders_png_previews_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            (prepared / "p1.musicxml").write_text("<score-partwise/>", encoding="utf-8")
            records = [
                {
                    "passage_id": "p1",
                    "composer": "Wolfgang Amadeus Mozart",
                    "composition_id": "cmp1",
                    "work_id": "w1",
                    "movement_id": "m1",
                    "recording_id": "r1",
                    "bar_start": 1,
                    "bar_end": 4,
                    "passage_generation_method": "fixed_4_bars",
                    "embedding_modality": "symbolic",
                    "status": "pending",
                    "error_message": "",
                    "content_hash": "",
                }
            ]
            status_map = {
                "p1": {
                    "status": "pending",
                    "error_message": "",
                    "content_hash": "h1",
                    "prepared_path": "prepared/p1.musicxml",
                }
            }
            binary = self._write_fake_musescore(root)

            review = export_review_passages(
                root / "review",
                records=records,
                status_map=status_map,
                prepared_root=prepared,
                musescore_binary=str(binary),
            )

            self.assertEqual(review["exported_rows"], 1)
            self.assertEqual(review["png_files"], 1)
            self.assertTrue((root / "review" / "prepared" / "p1.musicxml").is_file())
            self.assertTrue((root / "review" / "png" / "p1.png").is_file())
            with (root / "review" / "review_manifest.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["png_preview_paths"], "png/p1.png")

    def test_review_export_falls_back_to_pianoroll_when_musescore_is_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            prepared.mkdir()
            (prepared / "p1.musicxml").write_text(SYNTHETIC_MUSICXML, encoding="utf-8")
            records = [
                {
                    "passage_id": "p1",
                    "composer": "Wolfgang Amadeus Mozart",
                    "composition_id": "cmp1",
                    "work_id": "w1",
                    "movement_id": "m1",
                    "recording_id": "r1",
                    "bar_start": 1,
                    "bar_end": 4,
                    "passage_generation_method": "fixed_4_bars",
                    "embedding_modality": "symbolic",
                    "status": "pending",
                    "error_message": "",
                    "content_hash": "",
                }
            ]
            status_map = {
                "p1": {
                    "status": "pending",
                    "error_message": "",
                    "content_hash": "h1",
                    "prepared_path": "prepared/p1.musicxml",
                }
            }

            review = export_review_passages(
                root / "review",
                records=records,
                status_map=status_map,
                prepared_root=prepared,
                musescore_binary=str(root / "missing-mscore"),
                preview_renderer="auto",
            )

            self.assertEqual(review["exported_rows"], 1)
            self.assertEqual(review["png_files"], 1)
            self.assertEqual(review["musescore_version"], "")
            self.assertTrue((root / "review" / "png" / "p1.png").is_file())


class AnalysisCliTests(unittest.TestCase):
    def test_similarity_and_validation_reports_from_aligned_bundles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            embeddings = root / "embeddings"
            analysis = root / "analysis"
            corpus = root / "corpus"
            corpus.mkdir()
            config = {
                "corpus_root": str(corpus),
                "corpus_manifest": "manifest.csv",
                "score_root": "scores",
                "audio_root": "audio",
                "alignment_root": "alignments",
                "output_root": str(embeddings),
                "analysis_root": str(analysis),
                "log_root": str(root / "logs"),
                "temporary_root": str(root / "tmp"),
                "vendor_root": "vendor/clamp3",
                "checkpoint_path": "models/c2.pth",
                "saas_checkpoint_path": "models/saas.pth",
                "prompt_bank": "prompts/prompt_bank_v1.csv",
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            metadata = {
                "model": "CLaMP 3",
                "checkpoint_path": "checkpoint.pth",
                "checkpoint_hash": "test",
            }

            prompt_rows = [
                {
                    "prompt_id": "positive", "family": "matched_pair", "subfamily": "clarity",
                    "polarity": "positive", "embedding_row": 0, "embedding_dimension": 4,
                    "status": "success", "error_message": "",
                },
                {
                    "prompt_id": "negative", "family": "matched_pair", "subfamily": "clarity",
                    "polarity": "negative", "embedding_row": 1, "embedding_dimension": 4,
                    "status": "success", "error_message": "",
                },
            ]
            write_embedding_bundle(
                embeddings / "text", np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float), prompt_rows,
                matrix_filename="prompt_embeddings.npy", table_filename="prompt_embeddings.csv",
                fields=tuple(prompt_rows[0]), id_field="prompt_id",
                metadata={**metadata, "model_space": "c2"},
            )
            score_rows = [
                {
                    "score_id": "s1", "composer": "Wolfgang Amadeus Mozart", "period": "Classical",
                    "work_id": "w1", "movement_id": "m1", "source_path": "scores/s1.musicxml",
                    "embedding_row": 0, "embedding_dimension": 4, "status": "success", "error_message": "",
                }
            ]
            write_embedding_bundle(
                embeddings / "symbolic", np.array([[1, 0, 0, 0]], dtype=float), score_rows,
                matrix_filename="score_embeddings.npy", table_filename="score_embeddings.csv",
                fields=tuple(score_rows[0]), id_field="score_id",
                metadata={**metadata, "model_space": "c2"},
            )
            passage_rows = [
                {
                    "passage_id": "p1", "composer": "Frédéric Chopin", "period": "Romantic",
                    "work_id": "w2", "movement_id": "m2", "recording_id": "", "score_path": "scores/p1.musicxml",
                    "audio_path": "", "embedding_row": 0, "embedding_dimension": 4,
                    "status": "success", "error_message": "",
                }
            ]
            write_embedding_bundle(
                embeddings / "passages", np.array([[0, 1, 0, 0]], dtype=float), passage_rows,
                matrix_filename="passage_embeddings.npy", table_filename="passage_embeddings.csv",
                fields=tuple(passage_rows[0]), id_field="passage_id",
                metadata={**metadata, "model_space": "c2"},
            )
            audio_rows = [
                {
                    "recording_id": "r1", "audio_path": "", "embedding_row": "", "embedding_dimension": "",
                    "status": "unavailable", "error_message": "no local audio",
                }
            ]
            write_embedding_bundle(
                embeddings / "audio", np.empty((0, 4)), audio_rows,
                matrix_filename="recording_embeddings.npy", table_filename="recording_embeddings.csv",
                fields=tuple(audio_rows[0]), id_field="recording_id",
                metadata={**metadata, "model_space": "saas"},
            )

            project = Path(__file__).resolve().parents[1]
            similarity = subprocess.run(
                [sys.executable, str(project / "scripts" / "compute_similarities.py"), "--config", str(config_path)],
                cwd=project,
                capture_output=True,
                text=True,
            )
            self.assertEqual(similarity.returncode, 0, similarity.stderr)
            with (analysis / "similarities" / "prompt_to_score.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            with (analysis / "similarities" / "matched_prompt_axes.csv").open(newline="", encoding="utf-8") as handle:
                axes = list(csv.DictReader(handle))
            self.assertEqual({row["target_id"] for row in axes}, {"s1", "p1"})

            validation = subprocess.run(
                [sys.executable, str(project / "scripts" / "validate_embeddings.py"), "--config", str(config_path)],
                cwd=project,
                capture_output=True,
                text=True,
            )
            self.assertEqual(validation.returncode, 0, validation.stderr)
            report = json.loads((analysis / "reports" / "embedding_validation.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")


if __name__ == "__main__":
    unittest.main()
