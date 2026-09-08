import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from piano_clamp.embeddings import _preprocess_scores
from piano_clamp.features import extract_features, extract_musicxml_features
from piano_clamp.paths import repository_root


SYNTHETIC_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list>
    <score-part id="P1"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions><key><fifths>0</fifths><mode>major</mode></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><type>quarter</type></note>
      <note><chord/><pitch><step>E</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><type>quarter</type></note>
      <note><pitch><step>F</step><alter>1</alter><octave>4</octave></pitch><duration>1</duration><voice>1</voice><type>quarter</type><accidental>sharp</accidental></note>
      <note><rest/><duration>2</duration><voice>1</voice><type>half</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type><notations><ornaments><trill-mark/></ornaments></notations></note>
    </measure>
  </part>
</score-partwise>
"""


class MusicXMLFeatureTests(unittest.TestCase):
    def test_transparent_feature_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "passage.musicxml"
            path.write_text(SYNTHETIC_MUSICXML, encoding="utf-8")
            features = extract_musicxml_features(path)
        self.assertEqual(features["measure_count"], 2)
        self.assertEqual(features["pitched_note_count"], 4)
        self.assertEqual(features["note_event_count"], 5)
        self.assertAlmostEqual(features["rest_fraction"], 0.2)
        self.assertAlmostEqual(features["chord_tone_fraction"], 0.25)
        self.assertAlmostEqual(features["out_of_key_note_fraction"], 0.25)
        self.assertEqual(features["ornament_events_per_100_notes"], 25.0)

    def test_mxl_container_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "passage.mxl"
            container = """<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="score.musicxml" media-type="application/vnd.recordare.musicxml+xml"/></rootfiles></container>"""
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("META-INF/container.xml", container)
                archive.writestr("score.musicxml", SYNTHETIC_MUSICXML)
            features = extract_musicxml_features(path)
        self.assertEqual(features["measure_count"], 2)

    def test_extract_features_writes_external_table_and_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data" / "score.musicxml"
            source.parent.mkdir()
            source.write_text(SYNTHETIC_MUSICXML, encoding="utf-8")
            manifest = root / "manifest.csv"
            fields = (
                "passage_id", "composer", "work_title", "movement", "relative_path",
                "start_bar", "end_bar", "key", "meter",
            )
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "passage_id": "p1", "composer": "Chopin", "work_title": "Work",
                        "movement": "main", "relative_path": "score.musicxml", "start_bar": "1",
                        "end_bar": "2", "key": "C major", "meter": "4/4",
                    }
                )
            run_dir = root / "runs" / "test"
            output = extract_features(
                {
                    "paths": {
                        "manifest": str(manifest), "data_root": str(source.parent),
                        "run_dir": str(run_dir),
                    },
                    "models": {},
                    "prompt_files": [],
                }
            )
            self.assertTrue(output.is_file())
            self.assertTrue((run_dir / "inputs" / "score_data.lock.json").is_file())

    def test_official_symbolic_preprocessing_on_synthetic_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "score.musicxml"
            source.write_text(SYNTHETIC_MUSICXML, encoding="utf-8")
            workspace = root / "work"
            workspace.mkdir()
            model_inputs = _preprocess_scores(
                {"synthetic": source},
                vendor=repository_root() / "vendor" / "clamp3",
                workspace=workspace,
                log_path=root / "preprocessing.log",
            )
            generated = model_inputs / "synthetic.abc"
            self.assertTrue(generated.is_file())
            self.assertGreater(generated.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

