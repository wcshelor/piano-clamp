import csv
import json
import tempfile
import unittest
from pathlib import Path

from piano_clamp.data_loading import (
    CorpusManifestError,
    read_corpus_manifest,
    select_recording_records,
    select_symbolic_records,
)
from piano_clamp.passage_generation import (
    aligned_measure_coverage,
    aligned_time_range,
    generate_passage_records,
    materialize_symbolic_passage,
)
from piano_clamp.study import (
    StudyAuditError,
    build_clean_audio_render_manifest,
    build_clean_symbolic_manifest,
    build_study_readiness_report,
    freeze_study_manifest,
    write_clean_audio_render_manifest_bundle,
    write_clean_symbolic_manifest_bundle,
)


def corpus_row(**updates):
    row = {
        "passage_id": "cmp_demo",
        "composer": "Wolfgang Amadeus Mozart",
        "period": "Classical",
        "work_id": "wrk_demo",
        "movement_id": "mov_demo",
        "recording_id": "rec_demo",
        "score_path": "scores/demo.musicxml",
        "mxl_path": "",
        "audio_path": "audio/demo.wav",
        "midi_path": "",
        "alignment_path": "",
        "annotation_path": "",
        "bar_start": "1",
        "bar_end": "18",
        "score_format": "musicxml",
        "alignment_format": "",
        "rights_status": "licensed_for_research",
        "segment_type": "complete_movement",
        "annotation_type": "",
        "fully_aligned": "true",
        "eligible_for_clamp": "true",
        "availability_status": "available",
        "composition_id": "cmp_demo",
        "score_version_id": "scv_demo",
        "performance_id": "prf_demo",
        "alignment_id": "aln_demo",
        "audio_origin": "synthetic_rendering",
    }
    row.update(updates)
    return row


class CorpusLoadingTests(unittest.TestCase):
    def test_read_corpus_manifest_rejects_duplicate_passage_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.csv"
            rows = [corpus_row(), corpus_row(audio_path="audio/other.wav")]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "duplicate passage_id"):
                read_corpus_manifest(manifest)

    def test_select_recording_records_preserves_distinct_versions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifests").mkdir()
            (root / "audio").mkdir()
            (root / "audio" / "demo.wav").write_bytes(b"RIFFdemo")
            (root / "audio" / "demo_v2.wav").write_bytes(b"RIFFdemo2")
            with (root / "manifests" / "recordings.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("recording_id", "performer", "audio_path", "duration_seconds", "rights_status"),
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "recording_id": "rec_demo",
                            "performer": "Renderer A",
                            "audio_path": "audio/demo.wav",
                            "duration_seconds": "12.0",
                            "rights_status": "licensed_for_research",
                        },
                        {
                            "recording_id": "rec_demo_v2",
                            "performer": "Renderer B",
                            "audio_path": "audio/demo_v2.wav",
                            "duration_seconds": "12.5",
                            "rights_status": "licensed_for_research",
                        },
                    ]
                )

            records = select_recording_records(
                [
                    corpus_row(recording_id="rec_demo", audio_path="audio/demo.wav"),
                    corpus_row(
                        passage_id="cmp_demo_v2",
                        recording_id="rec_demo_v2",
                        audio_path="audio/demo_v2.wav",
                    ),
                ],
                root,
            )

            self.assertEqual([row["recording_id"] for row in records], ["rec_demo", "rec_demo_v2"])
            self.assertEqual({row["audio_origin"] for row in records}, {"synthetic_rendering"})
            self.assertTrue(all(row["status"] == "pending" for row in records))

    def test_read_corpus_manifest_rejects_conflicting_repeated_recording_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.csv"
            rows = [
                corpus_row(),
                corpus_row(
                    passage_id="cmp_demo_take2",
                    movement_id="mov_demo_alt",
                    audio_path="audio/other.wav",
                ),
            ]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(CorpusManifestError, "recording_id 'rec_demo' has conflicting registrations"):
                read_corpus_manifest(manifest)

    def test_select_symbolic_records_rejects_conflicting_score_versions_for_one_movement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifests").mkdir()
            rows = [
                corpus_row(score_path="scores/a.musicxml"),
                corpus_row(
                    passage_id="cmp_demo_v2",
                    recording_id="rec_demo_v2",
                    score_path="scores/b.musicxml",
                    audio_path="audio/demo_v2.wav",
                ),
            ]
            with self.assertRaisesRegex(CorpusManifestError, "movement_id 'mov_demo' has conflicting registrations"):
                select_symbolic_records(rows, root)


class PassageGenerationTests(unittest.TestCase):
    def test_fixed_windows_are_deterministic_and_drop_incomplete_tail(self):
        rows = [
            corpus_row(),
        ]

        generated = generate_passage_records(
            rows,
            corpus_root="/unused",
            method="fixed_bar_window",
            window_bars=8,
        )

        self.assertEqual([(row["bar_start"], row["bar_end"]) for row in generated], [(1, 8), (9, 16)])
        self.assertTrue(all(row["passage_generation_method"] == "fixed_8_bars" for row in generated))
        self.assertTrue(all(row["composition_id"] == "cmp_demo" for row in generated))
        self.assertTrue(all(row["audio_origin"] == "synthetic_rendering" for row in generated))
        self.assertEqual(len({row["passage_id"] for row in generated}), 2)

    def test_fixed_windows_support_overlap_via_smaller_stride(self):
        rows = [
            corpus_row(),
        ]

        generated = generate_passage_records(
            rows,
            corpus_root="/unused",
            method="fixed_bar_window",
            window_bars=8,
            stride_bars=4,
        )

        self.assertEqual(
            [(row["bar_start"], row["bar_end"]) for row in generated],
            [(1, 8), (5, 12), (9, 16)],
        )
        self.assertTrue(all(row["passage_generation_method"] == "fixed_8_bars_stride_4" for row in generated))

    def test_duplicate_source_rows_do_not_duplicate_passages(self):
        rows = [
            corpus_row(),
            corpus_row(passage_id="cmp_demo_alias"),
        ]

        generated = generate_passage_records(
            rows,
            corpus_root="/unused",
            method="fixed_bar_window",
            window_bars=16,
        )

        self.assertEqual(len(generated), 1)
        self.assertEqual(generated[0]["bar_start"], 1)
        self.assertEqual(generated[0]["bar_end"], 16)

    def test_performance_midi_materialization_uses_performance_source(self):
        try:
            import mido
        except ImportError:
            self.skipTest("mido is not installed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scores").mkdir()
            (root / "midi").mkdir()
            score = root / "scores" / "score.mid"
            performance = root / "midi" / "performance.mid"
            for path, velocity in ((score, 32), (performance, 96)):
                mid = mido.MidiFile()
                track = mido.MidiTrack()
                mid.tracks.append(track)
                track.append(mido.Message("note_on", note=60, velocity=velocity, time=0))
                track.append(mido.Message("note_off", note=60, velocity=0, time=480))
                track.append(mido.MetaMessage("end_of_track", time=0))
                mid.save(path)

            record = {
                "passage_id": "demo",
                "passage_generation_method": "complete_movement",
                "score_path": "scores/score.mid",
                "midi_path": "scores/score.mid",
                "performance_midi_path": "midi/performance.mid",
                "bar_start": 1,
                "bar_end": 1,
            }
            output = materialize_symbolic_passage(
                record,
                corpus_root=root,
                destination=root / "out",
                source_material="performance_midi",
            )

            self.assertEqual(output.resolve(), performance.resolve())

    def test_audio_time_range_requires_full_measure_coverage(self):
        record = {
            "recording_id": "rec_demo",
            "movement_id": "mov_demo",
            "bar_start": 1,
            "bar_end": 3,
        }
        events = {
            ("rec_demo", "mov_demo"): [
                {"score_measure": "1", "onset_seconds": "0.0", "offset_seconds": "1.0"},
                {"score_measure": "3", "onset_seconds": "2.0", "offset_seconds": "3.0"},
            ]
        }
        coverage = aligned_measure_coverage(record, events)
        self.assertEqual(coverage["missing_measures"], [2])
        self.assertFalse(coverage["fully_covered"])
        self.assertIsNone(aligned_time_range(record, events))


class StudyReadinessTests(unittest.TestCase):
    def _write_alignment_events(self, root: Path) -> Path:
        path = root / "alignment_events.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "recording_id",
                    "movement_id",
                    "score_measure",
                    "onset_seconds",
                    "offset_seconds",
                ),
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "recording_id": "rec_demo",
                        "movement_id": "mov_demo",
                        "score_measure": "1",
                        "onset_seconds": "0.0",
                        "offset_seconds": "1.0",
                    },
                    {
                        "recording_id": "rec_demo",
                        "movement_id": "mov_demo",
                        "score_measure": "2",
                        "onset_seconds": "1.0",
                        "offset_seconds": "2.0",
                    },
                    {
                        "recording_id": "rec_demo",
                        "movement_id": "mov_demo",
                        "score_measure": "3",
                        "onset_seconds": "2.0",
                        "offset_seconds": "3.0",
                    },
                ]
            )
        return path

    def test_study_readiness_reports_multi_interpretation_and_audio_window_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scores").mkdir()
            (root / "audio").mkdir()
            (root / "scores" / "demo.musicxml").write_text("<score-partwise/>", encoding="utf-8")
            (root / "audio" / "demo.wav").write_bytes(b"RIFFdemo")
            (root / "audio" / "demo_v2.wav").write_bytes(b"RIFFdemo2")
            alignment_path = self._write_alignment_events(root)
            rows = [
                corpus_row(bar_end="3"),
                corpus_row(
                    passage_id="cmp_demo_v2",
                    recording_id="rec_demo_v2",
                    audio_path="audio/demo_v2.wav",
                    rights_status="private_archive_only",
                    bar_end="3",
                ),
            ]
            report = build_study_readiness_report(
                rows,
                corpus_root=root,
                alignment_root=alignment_path,
                authorized_rights=("licensed_for_research",),
            )
            self.assertEqual(report["status"], "warning")
            self.assertEqual(report["audio_window_ready_rows"], 1)
            self.assertEqual(report["authorized_audio_rows"], 1)
            self.assertEqual(report["fully_aligned_audio_rows"], 1)
            self.assertEqual(report["multi_interpretation_compositions"][0]["composition_id"], "cmp_demo")
            self.assertEqual(report["multi_interpretation_compositions"][0]["recording_count"], 2)

    def test_freeze_manifest_writes_snapshot_and_readiness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scores").mkdir()
            (root / "audio").mkdir()
            (root / "scores" / "demo.musicxml").write_text("<score-partwise/>", encoding="utf-8")
            (root / "audio" / "demo.wav").write_bytes(b"RIFFdemo")
            manifest = root / "manifest.csv"
            rows = [corpus_row(bar_end="2")]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            self._write_alignment_events(root)
            config = {
                "corpus_manifest": str(manifest),
                "corpus_root": str(root),
                "alignment_root": str(root),
                "analysis_root": str(root / "analysis"),
                "repository_root": str(root),
                "configuration_path": str(root / "config.yaml"),
                "authorized_rights_statuses": ["licensed_for_research"],
            }
            destination = freeze_study_manifest(config, label="mozart-chopin-v1")
            self.assertTrue((destination / "piano_clamp_manifest.csv").is_file())
            self.assertTrue((destination / "study_readiness.json").is_file())
            metadata = json.loads((destination / "freeze_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["study_readiness_status"], "ok")

    def test_freeze_manifest_refuses_blocking_errors_without_override(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            rows = [
                corpus_row(),
                corpus_row(
                    passage_id="cmp_demo_take2",
                    movement_id="mov_demo_alt",
                    audio_path="audio/other.wav",
                ),
            ]
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            config = {
                "corpus_manifest": str(manifest),
                "corpus_root": str(root),
                "alignment_root": str(root),
                "analysis_root": str(root / "analysis"),
                "repository_root": str(root),
                "configuration_path": str(root / "config.yaml"),
                "authorized_rights_statuses": ["licensed_for_research"],
            }
            with self.assertRaisesRegex(CorpusManifestError, "recording_id 'rec_demo' has conflicting registrations"):
                freeze_study_manifest(config, label="bad-freeze")

    def test_clean_symbolic_manifest_excludes_conflicting_movements(self):
        rows = [
            corpus_row(movement_id="mov_ok", score_path="scores/a.musicxml", score_version_id="scv_a"),
            corpus_row(
                passage_id="cmp_demo_ok_2",
                movement_id="mov_ok_2",
                recording_id="rec_demo_2",
                score_path="scores/b.musicxml",
                audio_path="audio/demo_v2.wav",
                score_version_id="scv_b",
            ),
            corpus_row(
                passage_id="cmp_demo_conflict_1",
                movement_id="mov_conflict",
                recording_id="rec_demo_3",
                score_path="scores/c.musicxml",
                audio_path="audio/demo_v3.wav",
                score_version_id="scv_c",
            ),
            corpus_row(
                passage_id="cmp_demo_conflict_2",
                movement_id="mov_conflict",
                recording_id="rec_demo_4",
                score_path="scores/d.musicxml",
                audio_path="audio/demo_v4.wav",
                score_version_id="scv_d",
            ),
        ]

        report = build_clean_symbolic_manifest(rows)

        self.assertEqual(report["kept_row_count"], 2)
        self.assertEqual(report["excluded_row_count"], 2)
        self.assertEqual(report["excluded_movement_ids"], ["mov_conflict"])

    def test_write_clean_symbolic_manifest_bundle_writes_report_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "filtered"
            rows = [
                corpus_row(movement_id="mov_ok", score_path="scores/a.musicxml", score_version_id="scv_a"),
                corpus_row(
                    passage_id="cmp_demo_conflict_1",
                    movement_id="mov_conflict",
                    recording_id="rec_demo_3",
                    score_path="scores/c.musicxml",
                    audio_path="audio/demo_v3.wav",
                    score_version_id="scv_c",
                ),
                corpus_row(
                    passage_id="cmp_demo_conflict_2",
                    movement_id="mov_conflict",
                    recording_id="rec_demo_4",
                    score_path="scores/d.musicxml",
                    audio_path="audio/demo_v4.wav",
                    score_version_id="scv_d",
                ),
            ]
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)

            paths = write_clean_symbolic_manifest_bundle(
                destination,
                rows=rows,
                source_manifest_path=manifest,
            )

            self.assertTrue(paths["manifest_path"].is_file())
            self.assertTrue(paths["report_path"].is_file())
            with paths["manifest_path"].open(encoding="utf-8") as handle:
                exported = list(csv.DictReader(handle))
            self.assertEqual(len(exported), 1)
            self.assertEqual(exported[0]["movement_id"], "mov_ok")

    def test_clean_audio_render_manifest_reports_matches_and_mismatches(self):
        render_rows = [
            {
                "performance_id": "prf_match",
                "work_id": "wrk_match",
                "work_title": "Piano Sonata Match",
                "output_recording_id": "rec_match",
                "output_audio_path": "data/renders/spec-v1/rec_match.wav",
                "render_status": "complete",
                "qc_notes": "pass",
            },
            {
                "performance_id": "prf_new",
                "work_id": "wrk_new",
                "work_title": "Piano Sonata New",
                "output_recording_id": "rec_new",
                "output_audio_path": "data/renders/spec-v1/rec_new.wav",
                "render_status": "complete",
                "qc_notes": "pass",
            },
        ]
        export_rows = [
            corpus_row(recording_id="rec_match", audio_path="data/renders/spec-v1/rec_match.wav"),
            corpus_row(
                passage_id="cmp_old",
                recording_id="rec_old",
                performance_id="prf_old",
                audio_path="data/renders/spec-v1/rec_old.wav",
            ),
            corpus_row(
                passage_id="cmp_elsewhere",
                recording_id="rec_elsewhere",
                performance_id="prf_elsewhere",
                audio_path="data/renders/other-spec/rec_elsewhere.wav",
            ),
        ]

        report = build_clean_audio_render_manifest(render_rows, export_rows)

        self.assertEqual(report["matched_row_count"], 1)
        self.assertEqual(report["unmatched_render_recording_ids"], ["rec_new"])
        self.assertEqual(report["stale_export_recording_ids"], ["rec_old"])

    def test_write_clean_audio_render_manifest_bundle_writes_reconciliation_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "filtered_audio"
            render_manifest = root / "render.csv"
            export_manifest = root / "export.csv"
            render_rows = [
                {
                    "performance_id": "prf_match",
                    "work_id": "wrk_match",
                    "work_title": "Piano Sonata Match",
                    "output_recording_id": "rec_match",
                    "output_audio_path": "data/renders/spec-v1/rec_match.wav",
                    "render_status": "complete",
                    "qc_notes": "pass",
                },
                {
                    "performance_id": "prf_new",
                    "work_id": "wrk_new",
                    "work_title": "Piano Sonata New",
                    "output_recording_id": "rec_new",
                    "output_audio_path": "data/renders/spec-v1/rec_new.wav",
                    "render_status": "complete",
                    "qc_notes": "pass",
                },
            ]
            with render_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=render_rows[0])
                writer.writeheader()
                writer.writerows(render_rows)

            export_rows = [
                corpus_row(recording_id="rec_match", audio_path="data/renders/spec-v1/rec_match.wav"),
                corpus_row(
                    passage_id="cmp_old",
                    recording_id="rec_old",
                    performance_id="prf_old",
                    audio_path="data/renders/spec-v1/rec_old.wav",
                ),
            ]
            with export_manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=export_rows[0])
                writer.writeheader()
                writer.writerows(export_rows)

            paths = write_clean_audio_render_manifest_bundle(
                destination,
                render_rows=render_rows,
                export_rows=export_rows,
                render_manifest_path=render_manifest,
                export_manifest_path=export_manifest,
            )

            self.assertTrue(paths["manifest_path"].is_file())
            self.assertTrue(paths["report_path"].is_file())
            self.assertTrue(paths["unmatched_render_rows_path"].is_file())
            self.assertTrue(paths["stale_export_rows_path"].is_file())
            with paths["manifest_path"].open(encoding="utf-8") as handle:
                matched = list(csv.DictReader(handle))
            self.assertEqual(len(matched), 1)
            self.assertEqual(matched[0]["recording_id"], "rec_match")
            self.assertEqual(matched[0]["render_work_title"], "Piano Sonata Match")
            with paths["unmatched_render_rows_path"].open(encoding="utf-8") as handle:
                unmatched = list(csv.DictReader(handle))
            self.assertEqual([row["output_recording_id"] for row in unmatched], ["rec_new"])
            with paths["stale_export_rows_path"].open(encoding="utf-8") as handle:
                stale = list(csv.DictReader(handle))
            self.assertEqual([row["recording_id"] for row in stale], ["rec_old"])


if __name__ == "__main__":
    unittest.main()
