# Corpus passage preparation (legacy example)

This page documents the small, balanced PDMX symbolic pilot for the legacy
Chopin/Mozart example. For the larger canonical CPC score/audio workflow, use
the independent adapter documented in [cpc_adapter.md](cpc_adapter.md). Both
preparers are kept as explicitly scoped legacy dataset-preparation examples;
new dataset creation and corpus-scale asset rendering (including audio-from-MIDI)
belong in the external shared dataset layer, not in piano-clamp. piano-clamp
itself validates manifests, extracts features, creates embeddings, analyzes
embeddings, and renders review/result artifacts.

The primary score experiment uses a separate, immutable input bundle rather than
writing analysis-specific excerpts into `classical-performance-corpus`. The
preparer reads MusicXML directly from the verified PianoCoRe 1.0 raw-MIDI archive
and requires no network access or additional Python packages.

The fixed default policy is:

- exact Chopin and Wolfgang Amadeus Mozart composer identities;
- PianoCoRe tier A*, nonduplicate PDMX MusicXML scores;
- one centered 16-bar passage per selected work-unit;
- 16 work-units per composer;
- no horn-duo, two-piano, or song-title candidates;
- no more than two work-units from one parent composition; and
- deterministic diversity ordering with seed `20260819`.

Run:

```bash
python scripts/prepare_classical_corpus.py \
  --corpus-root /path/to/classical-performance-corpus \
  --output-root /path/to/chopin-mozart-pdmx-v1
```

The command refuses to replace an existing output directory. Choose a new
versioned path if the source archive, policy, or code changes. It verifies the
archive against the corpus-pinned MD5 before extraction and writes:

```text
chopin-mozart-pdmx-v1/
├── README.md
├── provenance.json
├── manifests/
│   ├── chopin_mozart.csv
│   └── selection.csv
└── scores/
    ├── chopin/
    └── mozart/
```

`selection.csv` retains the source archive member, PianoCoRe ID, original and
derived hashes, original measure labels, and the complete selection policy.
The nine-column `chopin_mozart.csv` is the only file consumed by the score
pipeline.

This controls passage length and score-source/encoding family, but it does not
eliminate repertoire, genre, historical-period, edition-quality, or model
pretraining confounds. The result remains a model-behavior proof of concept.
