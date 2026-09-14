# External data contract

piano-clamp is a consumer of external datasets and a producer of embeddings and
analysis outputs. It validates manifests, extracts features, creates embeddings,
analyzes embeddings, and renders review/result artifacts from existing dataset
manifests and assets. Dataset creation and source-asset rendering — including
audio-from-MIDI or corpus-scale MusicXML-to-audio — belong in the shared
dataset layer (for example `classical-performance-corpus`), not in piano-clamp.

All source data lives below an external dataset root (prefer
`PIANO_CLAMP_DATASETS_ROOT` or `PIANO_DATASETS_ROOT`; `PIANO_CLAMP_CORPUS_ROOT`
and the legacy `MUSIC_DATA_ROOT` remain supported as aliases); all generated
artifacts live below a run/output root (e.g. `CLAMP3_RUN_ROOT` or the
`output_root`/`embedding_store_root`/`analysis_root`/`log_root` configured in
`configs/embedding_config.yaml`). The repository neither writes to source data
nor contains copies of scores, MIDI, audio, embeddings, caches, or results
beyond locally generated adapter bundles.

## Score manifest

`configs/experiment_chopin_mozart.yaml` (legacy example; not the default) resolves
its manifest relative to the dataset root. In general, manifests are UTF-8 CSVs
with exactly these required columns. Extra columns may be present; the
recognized provenance columns described below are copied to the snapshot:

| Field | Meaning |
| --- | --- |
| `passage_id` | Unique, filename-safe identifier using letters, digits, `.`, `_`, or `-` |
| `composer` | Normalized composer label |
| `work_title` | Human-readable work title |
| `movement` | Movement or section label |
| `relative_path` | Score/MIDI path below the dataset root (`PIANO_CLAMP_DATASETS_ROOT`/`PIANO_DATASETS_ROOT`) |
| `start_bar` | One-based first bar represented by the file |
| `end_bar` | One-based final bar represented by the file |
| `key` | Curated key label |
| `meter` | Curated meter label |

Example:

```csv
passage_id,composer,work_title,movement,relative_path,start_bar,end_bar,key,meter
chopin_op10_no3_b001_016,Frédéric Chopin,Étude Op. 10 No. 3,main,scores/chopin/op10_no3_b001_016.mxl,1,16,E major,2/4
mozart_k545_i_b001_016,Wolfgang Amadeus Mozart,Sonata K. 545,I,scores/mozart/k545_i_b001_016.musicxml,1,16,C major,4/4
```

Each file must already represent the intended passage. `start_bar` and
`end_bar` document provenance; piano-clamp does not cut measures from
a complete score in this contract — passage windowing is handled by the
separate passage-generation layer when needed. Supported symbolic extensions are `.mxl`, `.musicxml`,
`.xml`, `.mid`, and `.midi`. Paths must be relative and cannot contain `..`.
The pipeline checks every file before model inference. Two passage IDs may not
reference the same file. For MusicXML/MXL, preflight parses the score and warns
when its measure count differs from the manifest bar span by more than the
configured pickup-measure tolerance.

The optional provenance columns `composition_id`, `work_id`,
`score_version_id`, `performance_id`, `recording_id`, `alignment_id`,
`audio_origin`, `source_dataset`, and `license` are copied into the normalized
snapshot when present. Score feature extraction also carries the composition,
work, score-version, and source-dataset fields into its table. In particular,
`composition_id` lets `analyze-features` keep movements from one parent
composition in the same inference group. Unrecognized extra columns remain
allowed but are not copied.

## Optional audio manifest

Audio uses the same nine-column contract and a separately configured
`paths.audio_manifest`. Its `relative_path` must end in `.wav` or `.mp3`, and
its `passage_id` should match the corresponding score passage when applicable.
Bar fields remain required so the excerpt boundaries are auditable.

The normalized score manifest, including recognized optional provenance, is copied to `manifest.snapshot.csv` in the run;
optional audio receives `audio_manifest.snapshot.csv`. The external source
repository is never changed.
