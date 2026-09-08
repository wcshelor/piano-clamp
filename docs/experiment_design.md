# Experiment design

## Question

This proof of concept asks whether pretrained CLaMP 3 C2 embeddings place
curated Chopin and Mozart piano passages differently relative to composer-name,
period, harmony, texture, lyricism, and ornamentation prompts.

## Reproducible unit of analysis

The passage is the unit of analysis. Passage files are prepared outside this
repository, identified by an immutable `passage_id`, and described by the
manifest contract. Sampling should balance composer, passage duration, genre,
and source quality where possible. Closely related passages from the same work
must not be treated as statistically independent without a work-level grouping
or sensitivity analysis.

## Pipeline

1. Validate and snapshot the external manifest and resolved configuration.
2. Extract the fixed MusicXML feature contract for an interpretable baseline.
3. Lock source bytes, manifest metadata, prompt wording, checkpoint identity,
   and the upstream commit against accidental same-run changes.
4. Run the official CLaMP 3 symbolic preprocessing: MusicXML/MXL becomes
   standard ABC and then interleaved ABC; MIDI becomes M3-compatible MTF.
5. Extract one global `(1, 768)` C2 vector per passage with `--get_global`.
6. Extract one C2 vector per fixed prompt.
7. Compute every passage–prompt cosine similarity and retain passage composer,
   prompt ID, and exact prompt text in a long CSV table.
8. Apply the preregistered work-level analysis in `docs/analysis_plan.md`.

Prompt wording is fixed in version-controlled YAML. Analyses should report
distributions and uncertainty, check robustness to prompt paraphrases, and
avoid selecting prompts after inspecting the desired result.

## C2 versus SAAS

C2 is the primary checkpoint and supplies both symbolic and text vectors for
the score–prompt comparison. SAAS is an optional checkpoint for audio. The two
models have 768-dimensional outputs, but dimensional equality does not make
their learned coordinate systems interchangeable. This repository never feeds
SAAS audio into the C2 prompt-similarity table. Any later audio–text analysis
must embed its text with SAAS and label that analysis as a separate checkpoint
space.

## Interpretation

Cosine similarity is evidence about this model's representation under a fixed
checkpoint, preprocessing path, corpus sample, and prompt wording. It is not a
direct measurement of musical style, composer intent, historical period, or a
listener's perception. Differences can reflect training data, metadata leakage,
encoding conventions, passage length, transcription practice, or prompt
sensitivity. Results are therefore descriptive model-behavior evidence and a
basis for validation—not proof that a passage intrinsically possesses a style.
