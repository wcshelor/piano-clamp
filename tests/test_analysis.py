import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from piano_clamp.analysis import analyze, analyze_features
from piano_clamp.provenance import ensure_input_lock


class AnalysisPipelineTests(unittest.TestCase):
    def test_preregistered_work_level_outputs_and_figure(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            tables = run_dir / "tables"
            scores = run_dir / "embeddings" / "scores"
            tables.mkdir(parents=True)
            scores.mkdir(parents=True)
            passages = [
                ("c1", "Chopin", "Chopin work 1"),
                ("c2", "Chopin", "Chopin work 2"),
                ("m1", "Mozart", "Mozart work 1"),
                ("m2", "Mozart", "Mozart work 2"),
            ]
            manifest_fields = (
                "passage_id", "composer", "work_title", "movement", "relative_path",
                "start_bar", "end_bar", "key", "meter",
            )
            with (run_dir / "manifest.snapshot.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=manifest_fields)
                writer.writeheader()
                for passage_id, composer, work in passages:
                    writer.writerow(
                        {
                            "passage_id": passage_id, "composer": composer, "work_title": work,
                            "movement": "main", "relative_path": f"{passage_id}.musicxml",
                            "start_bar": "1", "end_bar": "4", "key": "C major", "meter": "4/4",
                        }
                    )
            prompt_values = {
                "primary": [0.8, 0.7, 0.2, 0.3],
                "variant": [0.75, 0.65, 0.25, 0.35],
                "control": [0.1, 0.2, 0.1, 0.2],
            }
            similarity_rows = []
            for prompt_id, values in prompt_values.items():
                for (passage_id, composer, _), value in zip(passages, values):
                    similarity_rows.append(
                        {
                            "passage_id": passage_id, "composer": composer,
                            "prompt_id": prompt_id, "prompt_text": f"{prompt_id} prompt",
                            "cosine_similarity": value,
                        }
                    )
            pd.DataFrame(similarity_rows).to_csv(tables / "prompt_similarity.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "passage_id": passage_id, "composer": composer, "work_title": work,
                        "movement": "main", "measure_count": 4,
                        "note_density_per_measure": value,
                    }
                    for (passage_id, composer, work), value in zip(passages, [8.0, 7.0, 4.0, 5.0])
                ]
            ).to_csv(tables / "musicxml_features.csv", index=False)
            for index, (passage_id, _, _) in enumerate(passages):
                vector = np.zeros((1, 768), dtype=float)
                vector[0, index] = 1.0
                np.save(scores / f"{passage_id}.npy", vector)
            prompt_file = Path(temporary) / "prompts.yaml"
            prompt_file.write_text(
                yaml.safe_dump(
                    {
                        "prompts": [
                            {"id": "primary", "text": "primary prompt"},
                            {"id": "variant", "text": "variant prompt"},
                            {"id": "control", "text": "control prompt"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            ensure_input_lock(run_dir, "score_data", {"synthetic": "score_data"})
            config = {
                "experiment": {"composers": ["Chopin", "Mozart"]},
                "paths": {"run_dir": str(run_dir)},
                "prompt_files": [str(prompt_file)],
                "analysis": {
                    "contrast_order": ["Chopin", "Mozart"],
                    "primary_prompt_ids": ["primary"],
                    "robustness_pairs": [{"primary": "primary", "variant": "variant"}],
                    "bootstrap_iterations": 100,
                    "permutation_iterations": 100,
                    "random_seed": 7,
                    "require_features": True,
                },
            }
            feature_output = analyze_features(config)
            feature_contrasts = pd.read_csv(feature_output)
            self.assertIn("inference_group", feature_contrasts.columns)
            self.assertEqual(set(feature_contrasts["inference_group"]), {"work_title"})
            self.assertTrue((run_dir / "feature_analysis_plan.resolved.yaml").is_file())
            for lock_name in ("score_model", "prompt_data", "prompt_model"):
                ensure_input_lock(run_dir, lock_name, {"synthetic": lock_name})
            output = analyze(config)
            contrasts = pd.read_csv(output)
            primary = contrasts.loc[contrasts["prompt_id"] == "primary"].iloc[0]
            self.assertAlmostEqual(primary["mean_difference_a_minus_b"], 0.5)
            self.assertTrue((tables / "composer_feature_contrasts.csv").is_file())
            self.assertTrue((tables / "embedding_diagnostics.json").is_file())
            self.assertTrue((run_dir / "figures" / "primary_prompt_contrasts.png").is_file())
            self.assertTrue((run_dir / "analysis_plan.resolved.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
