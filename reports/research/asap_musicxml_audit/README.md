# ASAP Chopin MusicXML recovery audit

Audited upstream revision `afc815c75c42e83a79c03feb6da8a35e77d4c6b8`
of <https://github.com/fosfrancesco/asap-dataset>. All nine score-MIDI sources
reported as unsupported by the HPC feature extraction have upstream sibling
`xml_score.musicxml` files. `download_manifest.json` records checksums of the
18 downloaded XML/MIDI originals; raw scores are not committed. ASAP identifies
its dataset license as CC BY-NC-SA 4.0.

This was an upstream-only audit on macOS using music21 10.5.0, not a verification of
the corpus, prepared windows, embeddings, or features on the HPC. SSH reached
the configured host but failed authentication. The user elected to perform the
HPC steps themselves.

`upstream_audit.json` records parsed measure counts and 81 sample-window
comparisons (three positions at each of 4/8/16 bars, per work). MIDI windows used
music21's label-based `score.measures(start, end)`, as does the embedding
materializer. XML candidates used positional windows across all score parts,
as does the feature extractor. The conservative comparison required identical
pitched onsets, durations, grace events, tied-note continuity and sorted per-part
window durations, quantized to 1/960 quarter note. It ignored engraving and
staff/voice assignment in the resulting signature. Ambiguous matches were
rejected. No sample yielded a unique exact match. This does **not** establish
that the paired scores represent different compositions or cannot be aligned:
ornament realization, quantization, pickups and reconstructed MIDI bar grids
can defeat exact comparison. The repository specifies music21 >=9.1,<10, so
these importer results also require confirmation in the supported HPC environment.
It establishes that direct positional substitution
has not been validated.

Do not assign new XML features to old MIDI-derived passage IDs solely by work
name or bar numbers. Next: verify HPC source checksums/local XML availability,
then develop a score-time/note alignment against checksum-verified prepared
embedding windows and record a separate correspondence manifest. Preserve
existing embedding bundles and the 9,175 completed feature rows.

No embedding outputs were changed, so the HPC embedding inventory was not
regenerated.
