# Generated embeddings

This directory is the live output root for embedding bundles. Depending on the
stages you run, it may contain:

- `text/`
- `symbolic/`
- `audio/`
- `passages/`

Each bundle normally contains:

- a `.npy` matrix
- a CSV table with row metadata and status fields
- `metadata.json`

These live bundles are overwriteable with `--force`. They are not the durable
history of the project.

The durable history is the append-only `embedding_store/`:

- `embedding_store/runs/<run_id>/...` stores immutable snapshots of completed writes
- `embedding_store/registry.csv` stores one row per embedded item
- duplicates are skipped automatically
- passage rows retain method and derived window metadata such as `window_bars`

Use the live bundle for the current working state. Use the embedding store when
you need the historical record across reruns.

To import old live bundles that predate the store:

```bash
python scripts/backfill_embedding_store.py --config configs/embedding_config.yaml
```
