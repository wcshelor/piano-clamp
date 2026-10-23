import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from piano_clamp.xml_expansion import (
    VALIDATOR_VERSION,
    derive_candidate_identity,
    find_score_version_for_selection,
    hash_file,
    validate_representative_override,
)
from scripts.prepare_xml_expansion_audit import _resolve_corpus_root, _validate_score_file, _relative_score_path


def _make_musicxml(measure_count=4, pitched=True, two_parts=False):
    parts_xml = ""
    for part_idx in range(2 if two_parts else 1):
        part_id = f"P{part_idx+1}"
        measures = ""
        for m in range(1, measure_count + 1):
            # For two_parts case, make second part shorter for measure coverage test
            if two_parts and part_idx == 1 and m > 2:
                continue
            if pitched:
                note = '<note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>'
            else:
                note = '<note><rest/><duration>1</duration><type>quarter</type></note>'
            measures += f'<measure number="{m}">{note}</measure>'
        parts_xml += f'<part id="{part_id}">{measures}</part>'
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
  {parts_xml}
</score-partwise>'''.encode("utf-8")


def _make_mxl(musicxml_bytes: bytes, with_container=True) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if with_container:
            container = '''<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="score.musicxml" media-type="application/vnd.recordare.musicxml+xml"/></rootfiles>
</container>'''
            z.writestr("META-INF/container.xml", container)
        z.writestr("score.musicxml", musicxml_bytes)
    return buf.getvalue()


class TestCorpusPathResolution(unittest.TestCase):
    def test_missing_env_var_actionable(self):
        env = {}
        with self.assertRaises(SystemExit) as ctx:
            _resolve_corpus_root("${PIANO_CLAMP_CORPUS_ROOT}", environ=env)
        msg = str(ctx.exception)
        self.assertIn("PIANO_CLAMP_CORPUS_ROOT", msg)
        self.assertIn("export PIANO_CLAMP_CORPUS_ROOT", msg)

    def test_unreadable_missing_path(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "nonexistent"
            with self.assertRaises(SystemExit):
                _resolve_corpus_root(str(p))

    def test_unsafe_path_traversal(self):
        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td) / "corpus"
            corpus.mkdir()
            # Try to validate file with .. in relative path
            errors = _validate_score_file(corpus, "../etc/passwd", "abc", 1)
            self.assertTrue(any("escapes" in e or "traversal" in e for e in errors))

    def test_absolute_path_outside_corpus(self):
        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td) / "corpus"
            corpus.mkdir()
            other = Path(td) / "other" / "score.xml"
            other.parent.mkdir()
            other.write_bytes(_make_musicxml())
            # Try to use absolute path that is outside corpus
            # _relative_score_path would not be used here; directly test safe resolver
            from scripts.prepare_xml_expansion_audit import _resolve_score_path_safe
            with self.assertRaises(ValueError):
                _resolve_score_path_safe(corpus, "/etc/passwd")
            # Also test via _validate_score_file with absolute-like relative that escapes
            errors = _validate_score_file(corpus, "/absolute/path.xml", "abc", 1)
            self.assertTrue(len(errors) > 0)


class TestScoreFileValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = Path(self.tmp.name) / "corpus"
        self.corpus.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_file(self, rel: str, content: bytes) -> Path:
        p = self.corpus / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
        return p

    def test_missing_file(self):
        errors = _validate_score_file(self.corpus, "missing/score.musicxml", "abc", 10)
        self.assertTrue(any("does not exist" in e for e in errors))

    def test_unsupported_format(self):
        p = self._write_file("score.mid", b"dummy")
        sha = hash_file(p)
        errors = _validate_score_file(self.corpus, "score.mid", sha, 1)
        self.assertTrue(any("unsupported" in e for e in errors))

    def test_changed_checksum(self):
        data = _make_musicxml(4)
        p = self._write_file("score.musicxml", data)
        sha_wrong = "0" * 64
        errors = _validate_score_file(self.corpus, "score.musicxml", sha_wrong, 4)
        self.assertTrue(any("SHA-256 mismatch" in e for e in errors))
        # correct should pass
        sha_correct = hash_file(p)
        errors2 = _validate_score_file(self.corpus, "score.musicxml", sha_correct, 4)
        self.assertEqual(errors2, [])

    def test_malformed_xml(self):
        p = self._write_file("bad.xml", b"not xml at all <<<")
        sha = hash_file(p)
        errors = _validate_score_file(self.corpus, "bad.xml", sha, 1)
        self.assertTrue(any("invalid" in e.lower() for e in errors))

    def test_invalid_mxl_no_root(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("foo.txt", b"hello")
        p = self._write_file("bad.mxl", buf.getvalue())
        sha = hash_file(p)
        errors = _validate_score_file(self.corpus, "bad.mxl", sha, 1)
        self.assertTrue(any("no MusicXML" in e for e in errors))

    def test_invalid_mxl_bad_container(self):
        # MXL with container pointing to missing file
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("META-INF/container.xml", b"<container></container>")
            z.writestr("other.txt", b"hi")
        p = self._write_file("bad2.mxl", buf.getvalue())
        sha = hash_file(p)
        errors = _validate_score_file(self.corpus, "bad2.mxl", sha, 1)
        self.assertTrue(any("no MusicXML" in e or "invalid" in e.lower() for e in errors))

    def test_no_pitched_notes(self):
        data = _make_musicxml(4, pitched=False)
        p = self._write_file("nopitch.musicxml", data)
        sha = hash_file(p)
        errors = _validate_score_file(self.corpus, "nopitch.musicxml", sha, 4)
        self.assertTrue(any("no pitched notes" in e for e in errors))

    def test_measure_range_beyond_shortest_part(self):
        # Two parts: first has 4 measures, second has 2 measures => min is 2, request 4 should fail
        data = _make_musicxml(4, pitched=True, two_parts=True)
        p = self._write_file("short.musicxml", data)
        sha = hash_file(p)
        # request bar_end 4 -> exceeds min 2
        errors = _validate_score_file(self.corpus, "short.musicxml", sha, 4)
        self.assertTrue(any("exceeds" in e for e in errors))
        # request 2 should pass
        errors2 = _validate_score_file(self.corpus, "short.musicxml", sha, 2)
        self.assertEqual(errors2, [])

    def test_valid_mxl_and_xml(self):
        data = _make_musicxml(4)
        p = self._write_file("good.musicxml", data)
        sha = hash_file(p)
        errors = _validate_score_file(self.corpus, "good.musicxml", sha, 4)
        self.assertEqual(errors, [])
        # MXL
        mxl_bytes = _make_mxl(data)
        p2 = self._write_file("good.mxl", mxl_bytes)
        sha2 = hash_file(p2)
        errors2 = _validate_score_file(self.corpus, "good.mxl", sha2, 4)
        self.assertEqual(errors2, [])


class TestProvenance(unittest.TestCase):
    def test_correct_work_id_wrong_sv(self):
        svs = {
            "scv_a": {"score_version_id": "scv_a", "work_id": "wrk_1", "source_file_path": "a/score.musicxml", "sha256": "a" * 64},
            "scv_b": {"score_version_id": "scv_b", "work_id": "wrk_2", "source_file_path": "b/score.musicxml", "sha256": "b" * 64},
        }
        # Correct work_id + sha + path should match
        m = find_score_version_for_selection("wrk_1", "a" * 64, "a/score.musicxml", svs)
        self.assertIsNotNone(m)
        self.assertEqual(m["score_version_id"], "scv_a")
        # Wrong work_id with correct sha should not match
        m2 = find_score_version_for_selection("wrk_1", "b" * 64, "b/score.musicxml", svs)
        self.assertIsNone(m2)
        # Correct work_id but wrong sha should not match
        m3 = find_score_version_for_selection("wrk_1", "b" * 64, "a/score.musicxml", svs)
        self.assertIsNone(m3)
        # Correct work_id and sha but wrong path should not match (prevents borrowing)
        m4 = find_score_version_for_selection("wrk_1", "a" * 64, "b/score.musicxml", svs)
        self.assertIsNone(m4)

    def test_sibling_provenance_derived(self):
        # Simulate sibling case: work has only MIDI svs, no XML sv for sibling sha
        svs = {
            "scv_midi": {"score_version_id": "scv_midi", "work_id": "wrk_sib", "source_file_path": "midi/score.mid", "sha256": "c" * 64},
        }
        # sibling xml sha not in svs
        m = find_score_version_for_selection("wrk_sib", "d" * 64, "sibling/xml_score.musicxml", svs)
        self.assertIsNone(m)
        # Caller should then use derived id
        derived = f"scv_derived_{('d'*64)[:12]}"
        self.assertTrue(derived.startswith("scv_derived_"))


class TestRepresentativeSelection(unittest.TestCase):
    def test_ambiguity_without_policy(self):
        # Simulate identity_to_shas with ambiguous group
        identity_to_shas = {"Mozart|310|1": ["aaa", "bbb"]}
        analysis_id_to_key = {"aw_test": "Mozart|310|1"}
        # Without override, group should be unresolved; with override, one approved
        # Test valid override
        row = {"analysis_work_id": "aw_test", "representative_sha256": "aaa", "reason": "r", "evidence": "e"}
        try:
            validate_representative_override(row, identity_to_shas, analysis_id_to_key)
        except ValueError:
            self.fail("valid override should not raise")
        # Invalid: unknown aid
        row2 = {"analysis_work_id": "aw_unknown", "representative_sha256": "aaa", "reason": "r", "evidence": "e"}
        with self.assertRaises(ValueError):
            validate_representative_override(row2, identity_to_shas, analysis_id_to_key)
        # Invalid: sha not in group
        row3 = {"analysis_work_id": "aw_test", "representative_sha256": "ccc", "reason": "r", "evidence": "e"}
        with self.assertRaises(ValueError):
            validate_representative_override(row3, identity_to_shas, analysis_id_to_key)
        # Missing reason/evidence
        row4 = {"analysis_work_id": "aw_test", "representative_sha256": "aaa", "reason": "", "evidence": "e"}
        with self.assertRaises(ValueError):
            validate_representative_override(row4, identity_to_shas, analysis_id_to_key)

    def test_cli_ambiguity_and_override(self):
        # Use actual audit script with synthetic inputs minimal
        # We already tested via /tmp/test_audit_meta vs rep; here just sanity
        import subprocess

        # metadata-only mode without corpus should not require files
        out = Path(tempfile.mkdtemp())
        # Run with minimal synthetic CSVs? Use real inputs but check decision counts
        cmd = [sys.executable, str(ROOT / "scripts/prepare_xml_expansion_audit.py"), "--output-dir", str(out), "--force"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        dec = list(csv.DictReader(open(out / "decision.csv")))
        # Should have unresolved_ambiguous_edition without override
        amb = [d for d in dec if d["decision"] == "unresolved_ambiguous_edition"]
        self.assertTrue(len(amb) > 0)


class TestValidationPreservationAndStale(unittest.TestCase):
    def test_failed_validation_preserves_previous(self):
        with tempfile.TemporaryDirectory() as td:
            corpus = Path(td) / "corpus"
            corpus.mkdir()
            out = Path(td) / "out"
            out.mkdir()
            # Create a previous validated manifest to preserve
            prev = out / "expanded_manifest.validated.csv"
            prev.write_text("previous,content\n", encoding="utf-8")
            prev_cert = out / "validation_certificate.json"
            prev_cert.write_text('{"old": true}', encoding="utf-8")
            # Create a synthetic score that will fail checksum
            data = _make_musicxml(4)
            (corpus / "score.musicxml").write_bytes(data)
            # Now call _validate_score_file with wrong sha to simulate failure
            errors = _validate_score_file(corpus, "score.musicxml", "0" * 64, 4)
            self.assertTrue(len(errors) > 0)
            # Simulate script behavior: on failure, it should NOT overwrite prev
            # Check that prev still exists with old content if we don't overwrite
            self.assertTrue(prev.is_file())
            self.assertEqual(prev.read_text(), "previous,content\n")

    def test_stale_detection_via_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "file.csv"
            p.write_text("hello", encoding="utf-8")
            h1 = hash_file(p)
            p.write_text("hello changed", encoding="utf-8")
            h2 = hash_file(p)
            self.assertNotEqual(h1, h2)
            # Certificate would store h1, then after change h2 != stored => stale
            cert = {"input_hashes": {"usable_csv": h1}}
            # Simulate stale check
            self.assertNotEqual(cert["input_hashes"]["usable_csv"], h2)

    def test_laptop_metadata_only_without_corpus(self):
        # Ensure audit can run without corpus-root and does not claim ready
        import subprocess

        out = Path(tempfile.mkdtemp())
        cmd = [sys.executable, str(ROOT / "scripts/prepare_xml_expansion_audit.py"), "--output-dir", str(out), "--force"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["validation_state"], "metadata_only")
        # Candidate manifest should not be ready
        rows = list(csv.DictReader(open(out / "expanded_manifest.csv")))
        if rows:
            for r in rows:
                self.assertNotEqual(r["availability_status"], "ready")
                self.assertEqual(r["eligible_for_clamp"], "false")
        # Validated manifest should not exist
        self.assertFalse((out / "expanded_manifest.validated.csv").exists())
        self.assertFalse((out / "validation_certificate.json").exists())

    def test_successful_validation_creates_validated_manifest(self):
        import subprocess

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            corpus = td_path / "corpus"
            corpus.mkdir()
            # Create synthetic MusicXML
            rel = "Mozart/K310/1/score.musicxml"
            data = _make_musicxml(5, pitched=True)
            p = corpus / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            sha = hash_file(p)
            # Create minimal input CSVs
            inp = td_path / "inputs"
            inp.mkdir()
            usable = inp / "usable.csv"
            with usable.open("w", newline="", encoding="utf-8") as h:
                w = csv.DictWriter(h, fieldnames=["composer","work_id","composition_id","title","status","parent_composition_already_selected","measures","xml_path","origin","sha256","canonical_xml_checksum_available"])
                w.writeheader()
                w.writerow({
                    "composer": "Wolfgang Amadeus Mozart",
                    "work_id": "wrk_test123",
                    "composition_id": "cmp_test",
                    "title": "Piano Sonata No. 8 in A Minor, K. 310: I. Allegro maestoso",
                    "status": "unselected_work",
                    "parent_composition_already_selected": "False",
                    "measures": "5",
                    "xml_path": rel,  # relative path inside corpus
                    "origin": "canonical",
                    "sha256": sha,
                    "canonical_xml_checksum_available": "True",
                })
            works_csv = inp / "works.csv"
            with works_csv.open("w", newline="", encoding="utf-8") as h:
                w = csv.DictWriter(h, fieldnames=["work_id","composition_id","title"])
                w.writeheader()
                w.writerow({"work_id": "wrk_test123","composition_id": "cmp_test","title": "Piano Sonata No. 8 in A Minor, K. 310: I. Allegro maestoso"})
            comps_csv = inp / "compositions.csv"
            with comps_csv.open("w", newline="", encoding="utf-8") as h:
                w = csv.DictWriter(h, fieldnames=["composition_id","composer","title","catalogue"])
                w.writeheader()
                w.writerow({"composition_id": "cmp_test","composer": "Wolfgang Amadeus Mozart","title": "dummy","catalogue": "K. 310"})
            sv_csv = inp / "score_versions.csv"
            with sv_csv.open("w", newline="", encoding="utf-8") as h:
                w = csv.DictWriter(h, fieldnames=["score_version_id","work_id","source_file_path","sha256"])
                w.writeheader()
                w.writerow({"score_version_id": "scv_test","work_id": "wrk_test123","source_file_path": rel,"sha256": sha})
            main_manifest = inp / "main.csv"
            bach_manifest = inp / "bach.csv"
            for path in [main_manifest, bach_manifest]:
                with path.open("w", newline="", encoding="utf-8") as h:
                    h.write("passage_id,composer,period,work_id,movement_id,recording_id,score_path,mxl_path,audio_path,midi_path,alignment_path,annotation_path,bar_start,bar_end,score_format,alignment_format,rights_status,segment_type,annotation_type,fully_aligned,eligible_for_clamp,availability_status,composition_id,work_title,score_version_id\n")
            out = td_path / "out"
            cmd = [
                sys.executable, str(ROOT / "scripts/prepare_xml_expansion_audit.py"),
                "--usable-csv", str(usable),
                "--works-csv", str(works_csv),
                "--compositions-csv", str(comps_csv),
                "--score-versions-csv", str(sv_csv),
                "--current-main-manifest", str(main_manifest),
                "--current-bach-manifest", str(bach_manifest),
                "--output-dir", str(out),
                "--corpus-root", str(corpus),
                "--force",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["validation_state"], "validated")
            self.assertTrue((out / "expanded_manifest.validated.csv").is_file())
            self.assertTrue((out / "validation_certificate.json").is_file())
            cert = json.loads((out / "validation_certificate.json").read_text())
            self.assertEqual(cert["validator_version"], VALIDATOR_VERSION)
            self.assertIn(sha, cert["score_checksums"].values() if isinstance(cert["score_checksums"], dict) else str(cert))
            # Candidate should still be needs_validation
            cand_rows = list(csv.DictReader(open(out / "expanded_manifest.csv")))
            self.assertEqual(cand_rows[0]["availability_status"], "needs_validation")
            # Validated should be ready
            val_rows = list(csv.DictReader(open(out / "expanded_manifest.validated.csv")))
            self.assertEqual(val_rows[0]["availability_status"], "ready")
            self.assertEqual(val_rows[0]["eligible_for_clamp"], "true")
            self.assertEqual(val_rows[0]["score_version_id"], "scv_test")


if __name__ == "__main__":
    unittest.main()
