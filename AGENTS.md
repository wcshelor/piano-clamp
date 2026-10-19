# Piano CLaMP Agent Instructions

## HPC Dataset Path and Preflight Checks

The established HPC corpus directory is:

```text
/home/student/w/wshelor/share/datasets/classical-performance-corpus
```

For configurations that reference `${PIANO_CLAMP_CORPUS_ROOT}`, set:

```bash
export PIANO_CLAMP_CORPUS_ROOT=/home/student/w/wshelor/share/datasets/classical-performance-corpus
```

When creating or modifying dataset-consuming scripts and Slurm submission
wrappers, include an early preflight check by default. Check that environment
variables referenced by the selected configuration are set and non-empty,
and that the resolved corpus directory and selected manifest exist and are
readable. Fail before model loading or artifact writes with an actionable
error naming the missing variable or path. For a missing
`PIANO_CLAMP_CORPUS_ROOT` on the HPC, show the export command above.
Submission wrappers should check before calling `sbatch` and ensure required
variables are exported to the job environment.

Respect explicitly configured paths and supported aliases; do not require
this variable when the selected configuration does not use it. The HPC path
is deployment documentation, not a portable default for other machines.
Validate adapter/manifest composer coverage separately when required.

## Heavy HPC Artifacts

Embeddings and embedding-store snapshots are heavy HPC artifacts and are not
tracked directly in git. The committed source of truth for what currently
exists on the HPC is:

```text
reports/hpc/artifact_inventory/
```

Use `scripts/inventory_hpc_embeddings.py` to regenerate that inventory.

Whenever you implement, run, or recommend terminal commands that may create,
replace, delete, backfill, snapshot, or otherwise affect embedding artifacts on
the HPC, include an inventory refresh as the final step:

```bash
python scripts/inventory_hpc_embeddings.py \
  --embedding-root /home/student/w/wshelor/share/piano-clamp-runs/embeddings \
  --store-root /share/users/student/w/wshelor/piano-clamp/embedding_store \
  --output-dir reports/hpc/artifact_inventory
```

Then commit the generated lightweight inventory files, not the raw embeddings.
Read-only inspection commands do not need to regenerate the inventory, but any
workflow that might change embedding outputs should keep it current.

When planning HPC work, inspect the inventory before proposing reruns. Do not
assume text, score, passage, or audio embeddings are missing until the inventory
or live HPC paths show they are missing or stale.
