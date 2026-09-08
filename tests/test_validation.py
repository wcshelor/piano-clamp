import csv
import json
import tempfile
import unittest
from pathlib import Path

from piano_clamp.validation import DataValidationError, validate_data


class DataValidationTests(unittest.TestCase):
    def _config(self, root: Path, repeated: bool = False):
        data_root = root / "data"
        data_root.mkdir()
        rows = []
        for composer, prefix in (("Chopin", "c"), ("Mozart", "m")):
            for index in (1, 2):
                relative = f"{prefix}{1 if repeated else index}.musicxml"
                (data_root / relative).write_text("placeholder", encoding="utf-8")
                rows.append(
                    {
                        "passage_id": f"{prefix}{index}", "composer": composer,
                        "work_title": f"Work {index}", "movement": "main",
                        "relative_path": relative, "start_bar": "1", "end_bar": "4",
                        "key": "C major", "meter": "4/4",
                    }
                )
        manifest = root / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        return {
            "experiment": {"composers": ["Chopin", "Mozart"]},
            "data": {"passages_presegmented": True, "check_musicxml_parse": False},
            "paths": {
                "manifest": str(manifest), "data_root": str(data_root),
                "run_dir": str(root / "runs" / "test"),
            },
            "models": {},
            "prompt_files": [],
        }

    def test_balanced_two_work_manifest_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary))
            output = validate_data(config)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["work_counts"], {"Chopin": 2, "Mozart": 2})

    def test_reused_source_path_fails_with_written_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._config(root, repeated=True)
            with self.assertRaisesRegex(DataValidationError, "data validation failed"):
                validate_data(config)
            report = root / "runs" / "test" / "tables" / "data_validation.json"
            self.assertTrue(report.is_file())
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["status"], "error")


if __name__ == "__main__":
    unittest.main()
