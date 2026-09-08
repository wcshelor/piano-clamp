import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from piano_clamp.embeddings import score_prompts
from piano_clamp.similarity import (
    SimilarityError,
    build_similarity_rows,
    cosine_similarity_matrix,
    write_similarity_table,
)


class SimilarityTests(unittest.TestCase):
    def test_cosine_similarity_known_vectors(self):
        left = np.array([[1.0, 0.0], [1.0, 1.0]])
        right = np.array([[1.0, 0.0], [0.0, 1.0]])
        actual = cosine_similarity_matrix(left, right)
        expected = np.array([[1.0, 0.0], [2 ** -0.5, 2 ** -0.5]])
        np.testing.assert_allclose(actual, expected)

    def test_cosine_similarity_rejects_zero_vector(self):
        with self.assertRaisesRegex(SimilarityError, "zero-norm"):
            cosine_similarity_matrix(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]))

    def test_similarity_table_contract(self):
        passages = [{"passage_id": "p1", "composer": "Chopin"}]
        prompts = [{"prompt_id": "romantic", "prompt_text": "Romantic piano music."}]
        rows = build_similarity_rows(passages, prompts, np.array([[0.25]]))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "table.csv"
            write_similarity_table(rows, output)
            with output.open(newline="", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))
        self.assertEqual(
            list(written[0]),
            ["passage_id", "composer", "prompt_id", "prompt_text", "cosine_similarity"],
        )
        self.assertEqual(written[0]["passage_id"], "p1")

    def test_score_prompts_pipeline_uses_external_run_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "runs" / "test"
            score_dir = run_dir / "embeddings" / "scores"
            prompt_dir = run_dir / "embeddings" / "prompts"
            score_dir.mkdir(parents=True)
            prompt_dir.mkdir(parents=True)
            np.save(score_dir / "p1.npy", np.ones((1, 768)))
            np.save(prompt_dir / "romantic.npy", np.ones((1, 768)))

            manifest = root / "manifest.csv"
            source = root / "unused" / "p1.mxl"
            source.parent.mkdir()
            source.write_bytes(b"synthetic source identity")
            fields = (
                "passage_id",
                "composer",
                "work_title",
                "movement",
                "relative_path",
                "start_bar",
                "end_bar",
                "key",
                "meter",
            )
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "passage_id": "p1",
                        "composer": "Chopin",
                        "work_title": "Synthetic",
                        "movement": "main",
                        "relative_path": "unused/p1.mxl",
                        "start_bar": "1",
                        "end_bar": "4",
                        "key": "C major",
                        "meter": "4/4",
                    }
                )
            prompts = root / "prompts.yaml"
            prompts.write_text(
                yaml.safe_dump(
                    {"prompts": [{"id": "romantic", "text": "Romantic piano music."}]}
                ),
                encoding="utf-8",
            )
            config = {
                "paths": {
                    "run_dir": str(run_dir), "manifest": str(manifest),
                    "data_root": str(root),
                },
                "models": {
                    "c2": {"checkpoint": str(root / "c2.pth"), "modality": "symbolic_and_text"}
                },
                "clamp3": {"upstream_commit": "test"},
                "prompt_files": [str(prompts)],
            }
            output = score_prompts(config)
            with output.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertAlmostEqual(float(row["cosine_similarity"]), 1.0)
            self.assertTrue((run_dir / "manifest.snapshot.csv").is_file())


if __name__ == "__main__":
    unittest.main()
