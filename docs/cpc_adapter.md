# Canonical CPC adapter

Piano CLaMP builds its own adapter from the current
`classical-performance-corpus` (CPC) canonical tables. The adapter is
implemented in `piano_clamp.cpc_adapter`; it does not import code, manifests,
or generated artifacts from `mxl-clap`.

## Build

```bash
export PIANO_CLAMP_DATASETS_ROOT=/path/to/datasets  # or PIANO_DATASETS_ROOT / PIANO_CLAMP_CORPUS_ROOT (alias)
export PIANO_CLAMP_CORPUS_ROOT="$PIANO_CLAMP_DATASETS_ROOT/classical-performance-corpus"
python scripts/prepare_cpc_adapter.py \
  --corpus-root "$PIANO_CLAMP_CORPUS_ROOT" \
  --output-root data/cpc_adapter/cpc-2026-08-19.2-v1
# By default all eligible composers are retained. To filter, pass explicit values:
#   --composer Chopin --composer Mozart
# or use configs/experiment_chopin_mozart.yaml as a legacy example filter.
```

The default verifies every score and audio SHA-256 against CPC's canonical
tables. `--skip-hash-verification` is suitable for a quick path/schema audit,
but not for a published run. Outputs are written atomically, and an existing
adapter directory is never replaced; choose a new versioned directory when the
source release or adapter logic changes.

`--export-manifest` remains available only as a compatibility input for older
corpus exports. The normal path is to derive the study subset directly from
`data/canonical/` plus corpus release metadata.

The source corpus/dataset remains read-only. `adapter_root` in
`configs/embedding_config.yaml` points downstream commands at the generated
directory while all score and audio paths remain relative to the dataset root
(`PIANO_CLAMP_DATASETS_ROOT`/`PIANO_CLAMP_CORPUS_ROOT`). piano-clamp validates
manifests, extracts features, creates embeddings, analyzes embeddings, and
renders review/result artifacts; dataset creation and audio-from-MIDI rendering
belong in the dataset layer, not here.

## Outputs

```text
data/cpc_adapter/cpc-2026-08-19.2-v1/
├── piano_clamp_manifest.csv
├── alignment_events.csv
├── import_issues.csv
├── cohort_summary.csv
├── study_readiness.json
└── adapter_metadata.json
```

`piano_clamp_manifest.csv` is the normalized contract read by score, audio, and
passage embedding commands. It retains canonical composition, work, score,
performance, recording, and alignment IDs; source dataset and license text;
alignment quality; and `audio_origin`.

`alignment_events.csv` contains one contiguous, one-based performance-time
interval per score measure. CPC beat anchors are collapsed to measure
boundaries. CPC note anchors are interpolated through score time; this route
requires `music21`. Container duration is only an upper cap and never extends
an alignment beyond its final anchor.

For CPC release `cpc-2026-08-19.2` with all eligible composers (example: filtering to
Chopin/Mozart via `--composer Chopin --composer Mozart`), the validated
quick-audit output contains:

- 200 score/audio/alignment records;
- 41,994 measure timing rows with no duplicate or non-positive intervals;
- 108 real Chopin recordings across 24 parent compositions;
- one real Mozart recording and 91 synthetic Mozart recordings across 41
  parent compositions; and
- 16 excluded Mozart rows whose purported score path is a metadata CSV.

This is an example filtered output; the unfiltered adapter retains all eligible
composers by default.

## Analysis rules

Never use the combined audio rows as an unstratified composer comparison.
Separate `source_recording` and `synthetic_rendering`, and keep every row with
the same `composition_id` in one train/test or inferential group. These rules
are preserved in embedding metadata and summarized in `study_readiness.json`.

The adapter assigns `licensed_for_research` only when the CPC license string
contains an explicit CC-BY, MIT, or public-domain term. Other rows receive
`license_review_required` and are rejected by the default audio authorization
list until a human review changes the policy.
