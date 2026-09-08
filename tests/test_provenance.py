import tempfile
import unittest
import json
from pathlib import Path

from piano_clamp.embeddings import make_run_manifest
from piano_clamp.provenance import FingerprintError, ensure_input_lock, sha256_file


class ProvenanceTests(unittest.TestCase):
    def test_file_digest_is_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.txt"
            path.write_text("stable input\n", encoding="utf-8")
            self.assertEqual(
                sha256_file(path),
                "96df0717cefffafa5fc5b0ed0fdddd83a352b4d2344b4ef51f94ef169d2684f5",
            )

    def test_lock_allows_identical_resume_and_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = ensure_input_lock(temporary, "scores", {"value": 1})
            second = ensure_input_lock(temporary, "scores", {"value": 1})
            self.assertEqual(first, second)
            with self.assertRaisesRegex(FingerprintError, "choose a new"):
                ensure_input_lock(temporary, "scores", {"value": 2})

    def test_run_manifest_hashes_artifacts_and_excludes_temp(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            table = run_dir / "tables" / "result.csv"
            table.parent.mkdir(parents=True)
            table.write_text("value\n1\n", encoding="utf-8")
            cache = run_dir / "temp" / "cache.txt"
            cache.parent.mkdir()
            cache.write_text("transient", encoding="utf-8")
            config = {
                "paths": {"run_dir": str(run_dir)},
                "models": {
                    "c2": {"checkpoint": str(Path(temporary) / "c2.pth")},
                    "saas": {"checkpoint": str(Path(temporary) / "saas.pth")},
                },
                "prompt_files": [],
            }
            document = json.loads(make_run_manifest(config).read_text(encoding="utf-8"))
            artifacts = {item["path"]: item for item in document["artifacts"]}
            self.assertEqual(artifacts["tables/result.csv"]["sha256"], sha256_file(table))
            self.assertNotIn("temp/cache.txt", artifacts)


if __name__ == "__main__":
    unittest.main()
