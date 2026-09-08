# External data contract

All source data lives below `MUSIC_DATA_ROOT`; all generated artifacts live
below `CLAMP3_RUN_ROOT`. The repository neither writes to source data nor
contains copies of scores, MIDI, audio, embeddings, caches, or results.

## Score manifest

`configs/experiment_chopin_mozart.yaml` resolves its manifest relative to
`MUSIC_DATA_ROOT`. It is a UTF-8 CSV with exactly these required columns. Extra
columns may be present; the recognized provenance columns described below are
copied to the snapshot:

| Field | Meaning |
| --- | --- |
| `passage_id` | Unique, filename-safe identifier using letters, digits, `.`, `_`, or `-` |
| `composer` | Normalized composer label |
| `work_title` | Human-readable work title |
| `movement` | Movement or section label |
| `relative_path` | Score/MIDI path below `MUSIC_DATA_ROOT` |
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
`end_bar` document provenance; this proof of concept does not cut measures from
a complete score. Supported symbolic extensions are `.mxl`, `.musicxml`,
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
