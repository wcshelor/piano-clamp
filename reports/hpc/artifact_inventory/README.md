# HPC Artifact Inventory

This directory holds lightweight, committed inventories of heavyweight HPC
artifacts that are intentionally not stored in git.

Regenerate after embedding jobs on the HPC:

```bash
python scripts/inventory_hpc_embeddings.py \
  --embedding-root /home/student/w/wshelor/share/piano-clamp-runs/embeddings \
  --store-root /share/users/student/w/wshelor/piano-clamp/embedding_store \
  --output-dir reports/hpc/artifact_inventory
```

Commit the generated CSV, Markdown, and JSON files. Do not commit raw embedding
matrices or large source artifacts.
