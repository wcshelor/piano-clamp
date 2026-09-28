# Piano CLaMP Agent Instructions

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
