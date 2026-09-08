import csv
import tempfile
import unittest
from pathlib import Path

from piano_clamp.data import (
    MANIFEST_FIELDS,
    OPTIONAL_MANIFEST_FIELDS,
    ManifestError,
    read_manifest,
    resolve_source_files,
    write_manifest_snapshot,
)


def valid_row(**updates):
    row = {
        "passage_id": "chopin_001",
        "composer": "Frédéric Chopin",
        "work_title": "Étude",
        "movement": "main",
        "relative_path": "scores/chopin_001.mxl",
        "start_bar": "1",
        "end_bar": "16",
        "key": "E major",
        "meter": "2/4",
    }
    row.update(updates)
    return row


class ManifestContractTests(unittest.TestCase):
    def write_manifest(self, root: Path, row: dict[str, str]) -> Path:
        path = root / "manifest.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerow(row)
        return path

    def test_valid_manifest_and_source_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "scores" / "chopin_001.mxl"
            source.parent.mkdir()
            source.write_bytes(b"synthetic test placeholder")
            rows = read_manifest(self.write_manifest(root, valid_row()))
            resolved = resolve_source_files(rows, root)
            self.assertEqual(resolved["chopin_001"], source.resolve())

    def test_missing_required_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.csv"
            row = valid_row()
            fields = [field for field in MANIFEST_FIELDS if field != "meter"]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(row)
            with self.assertRaisesRegex(ManifestError, "missing columns: meter"):
                read_manifest(path)

    def test_parent_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_manifest(root, valid_row(relative_path="../private.mxl"))
            with self.assertRaisesRegex(ManifestError, "stay below MUSIC_DATA_ROOT"):
                read_manifest(path)

    def test_invalid_bar_range_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_manifest(root, valid_row(start_bar="20", end_bar="10"))
            with self.assertRaisesRegex(ManifestError, "end_bar precedes"):
                read_manifest(path)

    def test_recognized_provenance_survives_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "manifest.csv"
            row = valid_row(composition_id="cmp_demo", source_dataset="cpc")
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS + OPTIONAL_MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow(row)
            rows = read_manifest(path)
            self.assertEqual(rows[0]["composition_id"], "cmp_demo")
            snapshot = root / "manifest.snapshot.csv"
            write_manifest_snapshot(rows, snapshot)
            with snapshot.open(newline="", encoding="utf-8") as handle:
                copied = next(csv.DictReader(handle))
            self.assertEqual(copied["composition_id"], "cmp_demo")
            self.assertEqual(copied["source_dataset"], "cpc")


if __name__ == "__main__":
    unittest.main()
