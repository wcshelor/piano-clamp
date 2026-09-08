import csv
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from piano_clamp.musescore_conversion import convert_musescore_scores


VALID_MUSICXML = """<?xml version="1.0" encoding="UTF-8"?>
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
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type></note>
    </measure>
  </part>
</score-partwise>
"""


class MuseScoreConversionTests(unittest.TestCase):
    def _write_fake_musescore(self, root: Path) -> Path:
        binary = root / "fake_mscore"
        binary.write_text(
            """#!/usr/bin/env python3
import shutil
import sys
from pathlib import Path

VALID = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  <part id="P1"><measure number="1"><attributes><divisions>1</divisions><key><fifths>0</fifths><mode>major</mode></key><time><beats>4</beats><beat-type>4</beat-type></time></attributes><note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type></note></measure></part>
</score-partwise>
'''

if len(sys.argv) == 2 and sys.argv[1] == "--version":
    print("MuseScore 4.5.2")
    raise SystemExit(0)
if len(sys.argv) != 4 or sys.argv[1] != "-o":
    print("unexpected arguments", file=sys.stderr)
    raise SystemExit(2)
output = Path(sys.argv[2])
source = Path(sys.argv[3])
output.parent.mkdir(parents=True, exist_ok=True)
if source.stem == "broken":
    output.write_text("<not-musicxml/>", encoding="utf-8")
elif source.suffix.lower() == ".mscz":
    output.write_text(VALID, encoding="utf-8")
else:
    shutil.copyfile(source, output)
""",
            encoding="utf-8",
        )
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        return binary

    def test_conversion_writes_reports_and_validates_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = root / "output"
            report_root = root / "reports"
            input_root.mkdir()
            (input_root / "valid.mscx").write_text(VALID_MUSICXML, encoding="utf-8")
            (input_root / "broken.mscx").write_text("not used by the fake binary", encoding="utf-8")
            (input_root / "archive.mscz").write_text("not a real archive", encoding="utf-8")
            binary = self._write_fake_musescore(root)

            summary = convert_musescore_scores(
                input_root=input_root,
                output_root=output_root,
                report_root=report_root,
                musescore_binary=str(binary),
            )

            self.assertEqual(summary.source_count, 3)
            self.assertEqual(summary.converted_count, 2)
            self.assertEqual(summary.failed_count, 1)
            self.assertTrue((output_root / "valid.musicxml").is_file())
            self.assertTrue((output_root / "archive.musicxml").is_file())

            with (report_root / "musescore_conversion_report.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            by_name = {Path(row["source_path"]).name: row for row in rows}
            self.assertEqual(by_name["valid.mscx"]["status"], "success")
            self.assertEqual(by_name["archive.mscz"]["status"], "success")
            self.assertEqual(by_name["broken.mscx"]["status"], "failed")
            self.assertIn("score-partwise", by_name["broken.mscx"]["error"])

            report = json.loads((report_root / "musescore_conversion_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["musescore_version"], "MuseScore 4.5.2")
            self.assertEqual(report["converted_count"], 2)

    def test_existing_outputs_are_skipped_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = root / "output"
            report_root = root / "reports"
            input_root.mkdir()
            (input_root / "valid.mscx").write_text(VALID_MUSICXML, encoding="utf-8")
            binary = self._write_fake_musescore(root)

            first = convert_musescore_scores(
                input_root=input_root,
                output_root=output_root,
                report_root=report_root,
                musescore_binary=str(binary),
            )
            second = convert_musescore_scores(
                input_root=input_root,
                output_root=output_root,
                report_root=report_root,
                musescore_binary=str(binary),
            )

            self.assertEqual(first.converted_count, 1)
            self.assertEqual(second.skipped_count, 1)
            self.assertEqual(second.failed_count, 0)

    def test_cli_emits_json_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "input"
            output_root = root / "output"
            report_root = root / "reports"
            input_root.mkdir()
            (input_root / "valid.mscx").write_text(VALID_MUSICXML, encoding="utf-8")
            binary = self._write_fake_musescore(root)
            project = Path(__file__).resolve().parents[1]

            completed = subprocess.run(
                [
                    sys.executable,
                    str(project / "scripts" / "convert_musescore_scores.py"),
                    "--input-root",
                    str(input_root),
                    "--output-root",
                    str(output_root),
                    "--report-root",
                    str(report_root),
                    "--musescore-bin",
                    str(binary),
                    "--json",
                ],
                cwd=project,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["converted_count"], 1)
            self.assertEqual(payload["failed_count"], 0)


if __name__ == "__main__":
    unittest.main()
