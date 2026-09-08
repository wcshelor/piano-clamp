import tempfile
import unittest
from pathlib import Path

import yaml

from piano_clamp.paths import ConfigurationError, load_resolved_config, path_within
from piano_clamp.prompts import PromptError, load_prompts


def minimal_config():
    return {
        "experiment": {"run_name": "test_run"},
        "paths": {
            "data_root": "${MUSIC_DATA_ROOT}",
            "run_root": "${CLAMP3_RUN_ROOT}",
            "manifest": "manifests/test.csv",
        },
        "models": {
            "c2": {"checkpoint": "models/c2.pth"},
            "saas": {"checkpoint": "models/saas.pth"},
        },
        "prompt_files": ["configs/prompts.yaml"],
    }


class ConfigTests(unittest.TestCase):
    def test_environment_roots_and_repository_paths_are_resolved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(minimal_config()), encoding="utf-8")
            data = root / "external-data"
            runs = root / "external-runs"
            resolved = load_resolved_config(
                config_path,
                environ={"MUSIC_DATA_ROOT": str(data), "CLAMP3_RUN_ROOT": str(runs)},
                repo_root=root,
            )
            self.assertEqual(Path(resolved["paths"]["manifest"]), data / "manifests/test.csv")
            self.assertEqual(Path(resolved["paths"]["run_dir"]), runs / "test_run")
            self.assertEqual(Path(resolved["models"]["c2"]["checkpoint"]), root / "models/c2.pth")

    def test_missing_environment_variable_is_clear(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(minimal_config()), encoding="utf-8")
            with self.assertRaisesRegex(ConfigurationError, "MUSIC_DATA_ROOT is not set"):
                load_resolved_config(config_path, environ={}, repo_root=root)

    def test_path_resolution_rejects_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ConfigurationError, "escapes"):
                path_within(temporary, "../outside.csv", field="manifest")

    def test_run_directory_cannot_be_inside_data_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(minimal_config()), encoding="utf-8")
            data = root / "external-data"
            with self.assertRaisesRegex(ConfigurationError, "must not overlap"):
                load_resolved_config(
                    config_path,
                    environ={
                        "MUSIC_DATA_ROOT": str(data),
                        "CLAMP3_RUN_ROOT": str(data / "runs"),
                    },
                    repo_root=root,
                )

    def test_prompt_configuration_parsing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.yaml"
            path.write_text(
                yaml.safe_dump(
                    {"prompts": [{"id": "composer_chopin", "text": "A passage by Chopin."}]}
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                load_prompts([path]),
                [{"prompt_id": "composer_chopin", "prompt_text": "A passage by Chopin."}],
            )

    def test_duplicate_prompt_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "prompts": [
                            {"id": "same", "text": "First."},
                            {"id": "same", "text": "Second."},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PromptError, "duplicate prompt id"):
                load_prompts([path])


if __name__ == "__main__":
    unittest.main()
