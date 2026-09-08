import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path

import pandas as pd

from piano_clamp.cpc_adapter import build_cpc_adapter, supplement_cpc_adapter_with_render_manifest
from piano_clamp.data_loading import read_corpus_manifest
from piano_clamp.passage_generation import aligned_time_range, load_alignment_times


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CpcAdapterTests(unittest.TestCase):
    def test_canonical_tables_become_native_manifest_and_alignment_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus"
            canonical = corpus / "data" / "canonical"
            assets = corpus / "assets"
            canonical.mkdir(parents=True)
            (corpus / "manifests" / "releases").mkdir(parents=True)
            (corpus / "metadata").mkdir(parents=True)
            assets.mkdir()

            score = assets / "score.musicxml"
            audio = assets / "performance.wav"
            performance_midi = assets / "performance.mid"
            score.write_text("<score-partwise/>", encoding="utf-8")
            performance_midi.write_bytes(b"MThd\0\0\0\6\0\1\0\0\0`MTrk\0\0\0\4\0\xff/\0")
            with wave.open(str(audio), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(8_000)
                handle.writeframes(b"\0\0" * 24_000)

            pd.DataFrame(
                [
                    {
                        "composition_id": "cmp_demo",
                        "composer": "Wolfgang Amadeus Mozart",
                        "composer_normalized": "wolfgang amadeus mozart",
                        "title": "Canonical demo",
                    }
                ]
            ).to_csv(canonical / "compositions.csv", index=False)
            pd.DataFrame([{"work_id": "wrk_demo", "composition_id": "cmp_demo", "title": "Canonical demo"}]).to_csv(
                canonical / "works.csv", index=False
            )
            pd.DataFrame(
                [
                    {
                        "score_version_id": "scv_demo",
                        "work_id": "wrk_demo",
                        "source_file_path": "assets/score.musicxml",
                        "sha256": digest(score),
                    }
                ]
            ).to_csv(canonical / "score_versions.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "performance_id": "prf_demo",
                        "work_id": "wrk_demo",
                        "score_version_id": "scv_demo",
                    }
                ]
            ).to_csv(canonical / "performances.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "recording_id": "rec_demo_midi",
                        "performance_id": "prf_demo",
                        "media_type": "midi",
                        "rendering_kind": "refined_performance_midi",
                        "file_path": "assets/performance.mid",
                        "sha256": digest(performance_midi),
                        "source_dataset": "fixture",
                        "source_record_id": "src_demo_midi",
                        "source_record_ids": "src_demo_midi",
                        "qc_status": "pass",
                    },
                    {
                        "recording_id": "rec_demo",
                        "performance_id": "prf_demo",
                        "media_type": "audio",
                        "rendering_kind": "source_recording",
                        "file_path": "assets/performance.wav",
                        "sha256": digest(audio),
                        "source_dataset": "fixture",
                        "source_record_id": "src_demo",
                        "source_record_ids": "src_demo",
                        "qc_status": "pass",
                    }
                ]
            ).to_csv(canonical / "recordings.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "alignment_id": "aln_demo",
                        "work_id": "wrk_demo",
                        "score_version_id": "scv_demo",
                        "performance_id": "prf_demo",
                        "recording_id": "rec_demo",
                        "granularity": "beat",
                        "method": "source",
                        "quality_label": "source_exact",
                        "confidence": 1.0,
                        "sha256": "a" * 64,
                        "source_record_ids": "src_demo",
                        "qc_status": "pass",
                    }
                ]
            ).to_csv(canonical / "alignments.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "alignment_id": "aln_demo",
                        "measure_index": 1,
                        "performance_time_sec": 0.25,
                        "confidence": 1.0,
                    },
                    {
                        "alignment_id": "aln_demo",
                        "measure_index": 2,
                        "performance_time_sec": 1.25,
                        "confidence": 0.9,
                    },
                ]
            ).to_csv(canonical / "alignment_events.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "source_record_id": "src_demo",
                        "license": "CC-BY-4.0",
                    }
                ]
            ).to_csv(corpus / "metadata" / "source_records.csv", index=False)
            (corpus / "manifests" / "releases" / "cpc-test.json").write_text(
                json.dumps({"release": "cpc-test"}), encoding="utf-8"
            )

            output = root / "adapter"
            summary = build_cpc_adapter(corpus_root=corpus, output_root=output)

            self.assertEqual(summary.imported_rows, 1)
            self.assertEqual(summary.timing_rows, 2)
            rows = read_corpus_manifest(output / "piano_clamp_manifest.csv")
            self.assertEqual(rows[0]["composition_id"], "cmp_demo")
            self.assertEqual(rows[0]["bar_end"], "2")
            self.assertEqual(rows[0]["rights_status"], "licensed_for_research")
            self.assertEqual(rows[0]["performance_midi_path"], "assets/performance.mid")
            events = load_alignment_times(output)
            record = {
                "recording_id": "rec_demo",
                "movement_id": "wrk_demo",
                "bar_start": 1,
                "bar_end": 2,
            }
            self.assertEqual(aligned_time_range(record, events), (0.25, 2.25))
            readiness = json.loads((output / "study_readiness.json").read_text(encoding="utf-8"))
            self.assertEqual(readiness["status"], "ok")
            self.assertEqual(readiness["audio_window_ready_rows"], 1)
            self.assertEqual(readiness["required_grouping_key"], "composition_id")
            self.assertEqual(readiness["required_stratification_key"], "audio_origin")
            metadata = json.loads((output / "adapter_metadata.json").read_text(encoding="utf-8"))
            self.assertFalse(metadata["mxl_clap_dependency"])

    def test_supplement_adapter_adds_render_backed_rows_missing_from_primary_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus"
            canonical = corpus / "data" / "canonical"
            exports = corpus / "manifests" / "exports"
            renders = corpus / "manifests" / "rendering"
            assets = corpus / "assets"
            canonical.mkdir(parents=True)
            exports.mkdir(parents=True)
            renders.mkdir(parents=True)
            assets.mkdir()

            score = assets / "score.musicxml"
            score.write_text("<score-partwise/>", encoding="utf-8")
            audio_existing = assets / "performance.wav"
            audio_render = assets / "rendered.wav"
            source_midi = assets / "source.mid"
            source_midi.write_bytes(b"MThd\0\0\0\6\0\1\0\0\0`MTrk\0\0\0\4\0\xff/\0")
            for audio in (audio_existing, audio_render):
                with wave.open(str(audio), "wb") as handle:
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(8_000)
                    handle.writeframes(b"\0\0" * 24_000)

            pd.DataFrame([{"work_id": "wrk_demo", "title": "Canonical demo"}]).to_csv(
                canonical / "works.csv", index=False
            )
            pd.DataFrame(
                [
                    {"score_version_id": "scv_keep", "sha256": digest(score)},
                    {"score_version_id": "scv_render", "sha256": digest(score)},
                ]
            ).to_csv(canonical / "score_versions.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "recording_id": "rec_keep",
                        "sha256": digest(audio_existing),
                        "source_dataset": "fixture",
                        "media_type": "audio",
                        "qc_status": "pass",
                        "performance_id": "prf_keep",
                        "rendering_kind": "source_recording",
                        "file_path": "assets/performance.wav",
                        "source_record_id": "src_keep_audio",
                    }
                ]
            ).to_csv(canonical / "recordings.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "alignment_id": "aln_keep",
                        "granularity": "beat",
                        "method": "source",
                        "sha256": "a" * 64,
                        "work_id": "wrk_demo",
                        "score_version_id": "scv_keep",
                        "performance_id": "prf_keep",
                        "recording_id": "rec_keep",
                        "quality_label": "source_exact",
                        "confidence": 1.0,
                        "qc_status": "pass",
                        "source_record_ids": "src_keep_align",
                        "source_file_path": "data/canonical/alignment_events.csv",
                    },
                    {
                        "alignment_id": "aln_render",
                        "granularity": "beat",
                        "method": "source",
                        "sha256": "b" * 64,
                        "work_id": "wrk_demo",
                        "score_version_id": "scv_render",
                        "performance_id": "prf_render",
                        "recording_id": "rec_old",
                        "quality_label": "source_exact",
                        "confidence": 1.0,
                        "qc_status": "pass",
                        "source_record_ids": "src_render_align",
                        "source_file_path": "data/canonical/alignment_events.csv",
                    },
                ]
            ).to_csv(canonical / "alignments.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "alignment_id": "aln_keep",
                        "measure_index": 1,
                        "performance_time_sec": 0.25,
                        "confidence": 1.0,
                    },
                    {
                        "alignment_id": "aln_keep",
                        "measure_index": 2,
                        "performance_time_sec": 1.25,
                        "confidence": 1.0,
                    },
                    {
                        "alignment_id": "aln_render",
                        "measure_index": 1,
                        "performance_time_sec": 0.5,
                        "confidence": 1.0,
                    },
                    {
                        "alignment_id": "aln_render",
                        "measure_index": 2,
                        "performance_time_sec": 1.5,
                        "confidence": 1.0,
                    },
                ]
            ).to_csv(canonical / "alignment_events.csv", index=False)
            pd.DataFrame(
                [
                    {"composition_id": "cmp_demo", "composer": "Wolfgang Amadeus Mozart"},
                ]
            ).to_csv(canonical / "compositions.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "performance_id": "prf_keep",
                        "work_id": "wrk_demo",
                        "score_version_id": "scv_keep",
                        "source_dataset": "fixture",
                    },
                    {
                        "performance_id": "prf_render",
                        "work_id": "wrk_demo",
                        "score_version_id": "scv_render",
                        "source_dataset": "fixture",
                    },
                ]
            ).to_csv(canonical / "performances.csv", index=False)
            metadata = corpus / "metadata"
            metadata.mkdir()
            pd.DataFrame(
                [
                    {"source_record_id": "src_keep_audio", "license": "CC-BY-4.0"},
                    {"source_record_id": "src_keep_align", "license": "CC-BY-4.0"},
                    {"source_record_id": "src_render_align", "license": "CC-BY-4.0"},
                    {"source_record_id": "src_render", "license": "CC-BY-4.0"},
                ]
            ).to_csv(metadata / "source_records.csv", index=False)

            export_rows = [
                {
                    "export_row_id": "xpt_keep",
                    "corpus_release": "cpc-test",
                    "composer": "Wolfgang Amadeus Mozart",
                    "composition_id": "cmp_demo",
                    "work_id": "wrk_demo",
                    "score_version_id": "scv_keep",
                    "performance_id": "prf_keep",
                    "recording_id": "rec_keep",
                    "alignment_id": "aln_keep",
                    "score_path": "assets/score.musicxml",
                    "audio_path": "assets/performance.wav",
                    "alignment_path": "data/canonical/alignment_events.csv",
                    "audio_available": True,
                    "score_available": True,
                    "alignment_available": True,
                    "alignment_quality": "source_exact",
                    "alignment_confidence": 1.0,
                    "eligible_for_clap": True,
                    "synthetic_audio": False,
                    "source_record_ids": "src_keep_audio;src_keep_align",
                    "licenses": "CC-BY-4.0",
                    "original_source_dataset": "fixture",
                },
                {
                    "export_row_id": "xpt_render",
                    "corpus_release": "cpc-test",
                    "composer": "Wolfgang Amadeus Mozart",
                    "composition_id": "cmp_demo",
                    "work_id": "wrk_demo",
                    "score_version_id": "scv_render",
                    "performance_id": "prf_render",
                    "recording_id": "rec_old",
                    "alignment_id": "aln_render",
                    "score_path": "assets/score.musicxml",
                    "audio_path": "",
                    "alignment_path": "data/canonical/alignment_events.csv",
                    "audio_available": False,
                    "score_available": True,
                    "alignment_available": True,
                    "alignment_quality": "source_exact",
                    "alignment_confidence": 1.0,
                    "eligible_for_clap": False,
                    "synthetic_audio": True,
                    "source_record_ids": "src_render;src_render_align",
                    "licenses": "CC-BY-4.0",
                    "original_source_dataset": "fixture",
                },
            ]
            export = exports / "piano-clamp-chopin-mozart.csv"
            pd.DataFrame(export_rows).to_csv(export, index=False)
            (exports / "piano-clamp-schema.json").write_text(
                json.dumps({"schema_version": "1.1.0"}), encoding="utf-8"
            )

            adapter = root / "adapter"
            build_cpc_adapter(corpus_root=corpus, output_root=adapter, export_manifest=export)

            render_manifest = renders / "mozart-synthetic-v1.csv"
            pd.DataFrame(
                [
                    {
                        "performance_id": "prf_render",
                        "work_id": "wrk_demo",
                        "work_title": "Canonical demo",
                        "source_dataset": "fixture",
                        "performance_kind": "expressive",
                        "source_midi_recording_id": "rec_source_midi",
                        "source_midi_source_record_id": "src_render",
                        "source_midi_path": "assets/source.mid",
                        "source_midi_sha256": "c" * 64,
                        "score_version_id": "scv_render",
                        "score_path": "assets/score.musicxml",
                        "alignment_id": "aln_render",
                        "alignment_quality": "source_exact",
                        "last_aligned_performance_time_sec": "2.0",
                        "output_recording_id": "rec_render",
                        "output_audio_path": "assets/rendered.wav",
                        "render_spec_id": "render-v1",
                        "render_status": "complete",
                        "output_bytes": str(audio_render.stat().st_size),
                        "output_sha256": digest(audio_render),
                        "output_duration_sec": "3.0",
                        "output_sample_rate_hz": "8000",
                        "output_channels": "1",
                        "peak_dbfs": "-1.0",
                        "rms_dbfs": "-10.0",
                        "qc_notes": "pass",
                    }
                ]
            ).to_csv(render_manifest, index=False)

            supplemented = root / "adapter-supplemented"
            summary = supplement_cpc_adapter_with_render_manifest(
                corpus_root=corpus,
                adapter_root=adapter,
                render_manifest=render_manifest,
                output_root=supplemented,
                export_manifest=export,
            )

            self.assertEqual(summary.added_rows, 1)
            rows = read_corpus_manifest(supplemented / "piano_clamp_manifest.csv")
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["recording_id"] for row in rows}, {"rec_keep", "rec_render"})
            added = [row for row in rows if row["recording_id"] == "rec_render"][0]
            self.assertEqual(added["score_path"], "assets/score.musicxml")
            self.assertEqual(added["audio_origin"], "synthetic_rendering")
            self.assertEqual(added["performance_midi_path"], "assets/source.mid")
            events = load_alignment_times(supplemented)
            record = {
                "recording_id": "rec_render",
                "movement_id": "wrk_demo",
                "bar_start": 1,
                "bar_end": 2,
            }
            self.assertEqual(aligned_time_range(record, events), (0.5, 2.5))


if __name__ == "__main__":
    unittest.main()
