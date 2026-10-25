# Piano CLaMP Agent Instructions

## Minimize HPC Resource Requests

Always consider resource economy when planning, implementing, reviewing, or
recommending HPC jobs, including one-off terminal submissions. Request the
smallest practical GPU count, CPU count, memory allocation, and walltime that
can reliably complete the selected workload with a justified safety margin.
Do not copy generous resource limits from another job without checking them.

- Use CPU jobs for metadata audits, validation, MusicXML feature extraction,
  and other stages that do not benefit from a GPU. Separate these stages from
  GPU inference where practical.
- Before requesting GPU time, check the artifact inventory and existing caches,
  validate inputs on CPU, and estimate the actual pending work after reuse and
  deduplication. Prefer one GPU unless the implementation supports and benefits
  from multiple GPUs.
- Base requests on comparable completed jobs, measured peak memory and runtime,
  or a small representative pilot. State the evidence or assumptions behind
  proposed requests; do not present unmeasured estimates as requirements.
- Distinguish host RAM from GPU memory. Leave enough headroom for model loading,
  long inputs, and accounting uncertainty; avoid shrinking requests so far that
  repeated out-of-memory failures or timeouts waste more resources.
- Prefer checkpointed, resumable work and a realistic walltime over large
  speculative reservations. Reuse completed work after interruption.
- When a job queues, inspect its pending reason and actual GPU allocations
  before changing resources. Do not assume idle CPUs mean a GPU is available,
  and do not cancel/resubmit merely to try a smaller request when an appropriate
  pending-job update can preserve the existing submission.
- After execution, compare requested resources with observed use and use those
  measurements to size the next run.

See `docs/hpc_resource_requests.md` for the resource-sizing workflow. Resource
economy does not replace dataset preflight or embedding inventory requirements.

## HPC Dataset Path and Preflight Checks

The established HPC corpus directory is:

```text
/home/student/w/wshelor/share/datasets/classical-performance-corpus
```

The standard embedding configs load this path from `configs/machines/hpc.yaml`.
Keep dataset selection (`adapter_root`, `corpus_manifest`, `alignment_root`)
in run configs; `configs/paths.yaml` is restricted to machine settings and
must not redirect a run to another adapter. Use explicit manifest overrides
for ad hoc selections instead of cloning full embedding configs.

For other configurations that reference `${PIANO_CLAMP_CORPUS_ROOT}`, set:

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
