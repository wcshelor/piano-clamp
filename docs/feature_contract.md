# MusicXML feature contract

`extract-features` parses pre-segmented `.xml`, `.musicxml`, and `.mxl` files
with Python's standard XML and ZIP libraries. It does not rewrite scores and it
does not infer features from CLaMP embeddings. MIDI passages can be embedded by
C2 but are deliberately rejected by this feature command so modalities are not
silently mixed.

The table `tables/musicxml_features.csv` contains manifest identifiers followed
by these deterministic features:

| Feature | Definition |
| --- | --- |
| `measure_count` | Maximum measure count across parts |
| `note_event_count` | MusicXML `<note>` elements, including rests |
| `pitched_note_count` | Notes containing a parseable pitch |
| `rest_fraction` | Rest events divided by all note events |
| `grace_note_fraction` | Grace events divided by all note events |
| `note_density_per_measure` | Pitched notes divided by measures |
| `mean_midi_pitch`, `midi_pitch_std`, `midi_pitch_range` | Register and span in MIDI semitones |
| `pitch_class_entropy_bits` | Shannon entropy of pitch classes |
| `explicit_accidental_fraction` | Pitched notes carrying an explicit accidental |
| `out_of_key_note_fraction` | Pitch classes outside the natural major/minor scale implied by the first MusicXML key |
| `chord_tone_fraction` | Pitched notes marked with MusicXML `<chord/>` |
| `voice_count`, `staff_count` | Distinct part/voice and part/staff identifiers |
| `sounding_note_quarters` | Sum of non-chord, non-grace durations in quarter-note units; not wall-clock duration |
| `rhythmic_entropy_bits` | Shannon entropy of those quarter-note durations |
| `mean_abs_melodic_interval` | Mean absolute semitone motion within each part/voice/staff stream |
| `large_leap_fraction` | Melodic intervals of at least five semitones |
| `ornament_events_per_100_notes` | Trill, turn, mordent, and shake elements per 100 pitched notes |
| `key_fifths_xml`, `key_mode_xml` | First encoded key signature |
| `meter_beats_xml`, `meter_beat_type_xml` | First encoded time signature |

These are encoding-derived descriptors, not ground-truth perceptual features.
Missing or unusual MusicXML notation can change them. Polyphonic duration sums
and chord flags especially depend on exporter conventions. Their definitions
are fixed before analysis so they are not selected in response to results.

