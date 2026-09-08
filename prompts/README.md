# Prompt bank

`prompt_bank_v1.csv` is the canonical ordered prompt bank. `prompt_id` and
`prompt_text` are immutable within version 1; corrections require a new bank
version. The embedding script reads only rows whose `enabled` value is true and
stores every exact source row in run metadata.

Matched pairs use `family=matched_pair`, a shared `subfamily` axis name, and one
`positive` plus one `negative` pole. Axis scores are
`similarity(positive) - similarity(negative)`.
