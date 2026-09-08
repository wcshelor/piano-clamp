# Preregistered proof-of-concept analysis

This document and the `analysis` block in the experiment YAML fix the initial
analysis before real similarities are inspected.

## Units and contrast

The passage is the descriptive unit, but the work is the inferential unit.
Passage values are averaged within `composer × work_title`; bootstrap sampling
and label permutation then operate on works. The signed contrast is Frédéric
Chopin minus Wolfgang Amadeus Mozart.

## Primary outcomes

The eight original composer/descriptor prompts are primary outcomes. For each,
the analysis reports composer means at work level, their difference, a 95%
work-clustered bootstrap interval, Hedges' g, and a two-sided work-label
permutation p-value. Benjamini–Hochberg correction is applied only across these
eight preregistered primary outcomes.

## Robustness and controls

Four prompt paraphrases are compared with their primary prompt by effect size
and direction. A neutral piano prompt and an unrelated ocean-recording prompt
are descriptive negative controls and are excluded from the primary FDR family.
They do not prove absence of leakage or bias.

## MusicXML features

Every numeric feature with at least two work-level values per composer receives the same
work-level contrast, bootstrap interval, effect size, permutation test, and a
separate feature-family FDR correction. Features without adequate coverage are
retained in the output with a `skipped` status and reason. These comparisons are
exploratory even though their computation is fixed in advance.

`analyze-features` runs this feature family independently of all CLaMP model and
prompt artifacts. Its random stream uses the configured analysis seed plus one,
so its output is identical whether it is run by itself or as part of `analyze`.
The inference group is `composition_id` when that column is present in both the
manifest and feature table; otherwise it is `work_title`. This keeps related
movements together when an expanded corpus supplies parent-composition IDs.

## Diagnostics and fixed computation

The random seed is `20260819`. The default uses 5,000 bootstrap samples and up
to 10,000 work-label permutations (exact enumeration when smaller). Diagnostics
check finite vectors, norms, exact duplicate embeddings, and sampled within- and
between-composer cosine similarities. Model outputs remain evidence about CLaMP
3 behavior rather than direct measurements of historical or perceptual style.
