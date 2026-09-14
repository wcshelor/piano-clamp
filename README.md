# Piano CLaMP

Test. Piano CLaMP is a reproducible embedding-and-analysis pipeline for a focused
computational musicology project on piano music. It does not train or fine-tune models. It stages symbolic scores,
text prompts, and optionally authorized local audio through pinned CLaMP 3
checkpoints, writes provenance-rich embedding bundles, and keeps an append-only
registry of what has already been embedded.

The project’s main goals are:

- embed symbolic scores, prompt texts, and score/audio passages in a controlled way
- preserve enough metadata to distinguish composer, work, movement, recording, and passage windowing
- support incremental HPC runs without losing prior work
- make later similarity analysis and manual review possible without re-deriving everything

This repository is the active home of that workflow. It reads
`classical-performance-corpus` as an external input and does not write back into
that corpus.

## What The Repo Does

There are four primary embedding products:

- `text`: embeddings for the canonical prompt bank
- `symbolic`: one global C2 embedding per supported unique score file
- `audio`: optional SAAS chunk embeddings plus pooled recording embeddings for authorized local audio
- `passages`: symbolic or aligned-audio window embeddings, including fixed 4/8/16-bar windows

There are also three important operational products:

- `embedding_store`: immutable run snapshots plus a registry of all embedded items
- `analysis`: validation reports, similarity tables, nearest-neighbor outputs, frozen manifests
- `logs`: stage logs, Slurm logs, and other runtime diagnostics

The workflow is built around the upstream CLaMP 3 code in `vendor/clamp3`, but
this repo wraps it with deterministic staging, metadata capture, deduplication,
resume behavior, and study-specific utilities.

## Current Model Roles

- `CLaMP 3 C2` is the required model for symbolic scores, symbolic passages, and text prompts.
- `CLaMP 3 SAAS` is optional and intended for audio and aligned-audio passages.

Both emit 768-dimensional vectors, but they are different embedding spaces.
Never compare C2 vectors directly against SAAS vectors.

## Repo Map

These are the directories that matter most:

| Path | Purpose |
|---|---|
| `configs/` | Portable defaults plus local path overrides |
| `data/cpc_adapter/` | Locally generated adapter bundles derived from CPC |
| `docs/` | Design notes, contracts, and adapter/data documentation |
| `embedding_store/` | Append-only snapshot store and registry |
| `analysis/` | Generated reports, similarity outputs, frozen manifests |
| `logs/` | Generated runtime and Slurm logs |
| `jobs/` | Slurm batch scripts for common HPC runs |
| `models/` | Local checkpoint locations |
| `prompts/` | Versioned canonical prompt bank |
| `scripts/` | Main operational entry points |
| `src/piano_clamp/` | Core pipeline implementation |
| `tests/` | Regression tests for pipeline behavior |
| `vendor/clamp3/` | Pinned upstream CLaMP 3 code |

The most important Python modules are:

- `src/piano_clamp/data_loading.py`: config loading, path resolution, manifest/prompt loading
- `src/piano_clamp/clamp_backend.py`: controlled interface to upstream CLaMP extraction
- `src/piano_clamp/passage_generation.py`: passage/window generation and aligned audio extraction
- `src/piano_clamp/embedding_io.py`: matrix/table/metadata writing and overwrite guards
- `src/piano_clamp/embedding_store.py`: immutable snapshotting and registry updates
- `src/piano_clamp/metadata.py`: run metadata, run IDs, log path naming
- `src/piano_clamp/passage_artifacts.py`: failure tables and passage review exports

## Configuration

The default portable config is [configs/embedding_config.yaml](configs/embedding_config.yaml).
It assumes:

- a read-only external corpus root
- a local CPC adapter under `data/cpc_adapter/`
- local outputs for embeddings, logs, analysis, and temp files

The normal local override mechanism is [configs/paths.yaml](configs/paths.yaml).
`load_pipeline_config()` auto-merges that file when you pass
`configs/embedding_config.yaml`, so in normal repo usage you should not need to
manually export `PIANO_CLAMP_CORPUS_ROOT` every time.

In practice:

- commit portable defaults in `configs/embedding_config.yaml`
- keep machine-specific absolute paths in ignored `configs/paths.yaml`
- pass `--config configs/embedding_config.yaml` to scripts

Key config fields:

- `corpus_root`
- `adapter_root`
- `corpus_manifest`
- `score_root`
- `audio_root`
- `alignment_root`
- `output_root`
- `embedding_store_root`
- `analysis_root`
- `log_root`
- `temporary_root`
- `checkpoint_path`
- `saas_checkpoint_path`
- `prompt_bank`
- `authorized_rights_statuses`

## Data Model

The study is source-centric. Different performances, renders, or
interpretations of the same piece should remain distinct records rather than
silent replacements.

Important identifiers and fields:

- `composer`
- `composition_id`
- `work_id`
- `movement_id`
- `recording_id`
- `audio_origin`
- `score_id`
- `passage_id`
- `passage_generation_method`
- `window_bars`
- `start_bar`
- `end_bar`

For passage work, this matters especially:

- duplicate score/audio content should not create duplicate registry entries
- distinct performances of the same piece should remain separate if their source identity differs
- 4-, 8-, and 16-bar windows must remain distinguishable in tables and registry rows

## Embedding Store

The live bundle under `output_root` is overwriteable with `--force`. The
append-only store is the durable record.

Store behavior:

- each successful bundle is copied to `embedding_store/runs/<run_id>/...`
- `embedding_store/registry.csv` stores one row per embedding item
- duplicates are skipped based on identity/hash rules
- passage rows retain method and derived window metadata
- immutable snapshots let you rerun live outputs without losing prior runs

If you have older bundles that predate the store, import them with:

```bash
python scripts/backfill_embedding_store.py --config configs/embedding_config.yaml
```

## Core Workflows

### 1. Prepare the adapter

Build the local CPC adapter bundle without editing the external corpus:

```bash
python scripts/prepare_cpc_adapter.py \
  --corpus-root /path/to/classical-performance-corpus \
  --output-root data/cpc_adapter/cpc-2026-08-19.2-v1
```

See [docs/cpc_adapter.md](docs/cpc_adapter.md) for adapter details.

### 2. Check the environment

```bash
python scripts/check_installation.py --config configs/embedding_config.yaml --json
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

If `--device cuda` is requested, `torch.cuda.is_available()` must be `True`.

### 3. Embed text prompts

```bash
python scripts/embed_text_prompts.py \
  --config configs/embedding_config.yaml \
  --device cuda
```

Optional SAAS-space text embeddings for audio-text experiments:

```bash
python scripts/embed_text_prompts.py \
  --config configs/embedding_config.yaml \
  --device cuda \
  --model-space saas
```

### 4. Embed symbolic scores

```bash
python scripts/embed_symbolic_scores.py \
  --config configs/embedding_config.yaml \
  --device cuda \
  --composer Chopin
```

### 5. Embed passages

Score-symbolic passages, including fixed 4/8/16-bar windows:

```bash
python scripts/embed_passages.py \
  --config configs/embedding_config.yaml \
  --device cuda \
  --modality symbolic \
  --source-material score \
  --composer Chopin \
  --mode 4 \
  --mode 8 \
  --mode 16 \
  --overlap-half \
  --status-every 10
```

Performance-MIDI passages use CLaMP's symbolic space but select the
performance MIDI source and crop fixed windows by aligned performance time, so
timing, velocity, and controller events remain in the MIDI given to CLaMP:

```bash
python scripts/embed_passages.py \
  --config configs/embedding_config.yaml \
  --device cuda \
  --modality symbolic \
  --source-material performance_midi \
  --composer Chopin \
  --mode 4 \
  --mode 8 \
  --mode 16 \
  --overlap-half
```

Audio passages use the same script with `--source-material audio` and require:

- `--modality audio`
- authorized rights status
- local audio path
- alignment events covering the requested measure range

Score passages write to `embeddings/passages`, performance MIDI passages write
to `embeddings/passages/performance_midi`, and aligned audio passages write to
`embeddings/passages/audio`.

### 6. Validate and analyze

```bash
python scripts/validate_embeddings.py --config configs/embedding_config.yaml
python scripts/build_embedding_index.py --config configs/embedding_config.yaml
python scripts/compute_similarities.py --config configs/embedding_config.yaml --top-k 20
```

### 7. Export passage review artifacts

Use this when you want to manually inspect generated windows as viewable score
artifacts:

```bash
python scripts/export_passage_review.py \
  --config configs/embedding_config.yaml \
  --composer Chopin \
  --mode 4 \
  --mode 8 \
  --mode 16 \
  --overlap-half \
  --source-material score \
  --status failed \
  --preview-renderer pianoroll
```

This exports cached prepared symbolic passage files plus a manifest so you can
spot bad windowing or malformed examples before trusting downstream analysis.
`--preview-renderer pianoroll` uses the built-in `music21` + `matplotlib`
preview path and avoids any external MuseScore dependency.

For the Mozart review batch job in
`piano-clamp-runs/jobs/queue_mozart_passage_review.sbatch`, the MuseScore CLI
path is controlled by `MUSESCORE_BIN` and defaults to `mscore`. On clusters,
prefer passing an explicit path with `sbatch --export=ALL,MUSESCORE_BIN=/path/to/mscore ...`
instead of relying on the compute-node `PATH`.

## Resume And Overwrite Semantics

These are easy to confuse:

- `--force` overwrites the final live bundle files in `output_root`
- `--force` does not delete resume caches
- `--reset-resume` discards the passage resume cache under `.resume`
- passage preprocessing/materialization caches can be reused across reruns

For passage jobs, the `.resume` directory keeps:

- per-record status
- prepared passage cache
- vector cache
- failure rows

That means an interrupted run can usually resume without redoing the expensive
materialization step.

## Logging

Stage logs now use per-run filenames instead of one append-forever file. The
run metadata records the exact `log_path` for that run.

Generated logs usually include:

- stage logs under `log_root`
- Slurm stdout/stderr files under `logs/`
- passage preparation and chunk progress
- upstream CLaMP diagnostics

The most useful runtime progress behavior today is in `embed_passages.py`,
which reports:

- planned passage counts by method
- resume-cache status
- preparation progress
- chunk-level embedding progress
- success/failure counts and ETA

## HPC Usage

For unattended cluster work, prefer `sbatch` jobs in [jobs/](jobs/).
Current useful examples include:

- [jobs/embed_passages_chopin.slurm](jobs/embed_passages_chopin.slurm)
- [jobs/embed_text_prompts.slurm](jobs/embed_text_prompts.slurm)

Typical workflow:

1. submit with `sbatch jobs/<name>.slurm`
2. check active jobs with `squeue -u <user>`
3. check finished history with `sacct`
4. inspect the actual log file in `logs/slurm-<jobname>-<jobid>.out`

Remember:

- `squeue` only shows pending/running jobs
- finished jobs disappear from `squeue`
- use the real numeric job ID, not a literal `<jobid>` placeholder

## MuseScore On HPC

`scripts/export_passage_review.py` needs a working MuseScore CLI binary to
render review artifacts. The most common cluster failure mode is that the batch
job starts correctly and then exits almost immediately because the binary is
unavailable or incompatible on the compute node.

Observed on August 26, 2026:

- job `12810835` failed with `error: MuseScore CLI is not available: mscore`
- job `12811139` found `MUSESCORE_BIN`, but the MuseScore 4 AppImage failed on
  the GPU node with `GLIBC_2.34 not found`
- the same AppImage also reported `libasound.so.2: cannot open shared object file`
- `srun -p gpu --time=00:05:00 --mem=2G bash -lc 'ldd --version | head -1'`
  reported `glibc 2.28`

Implications:

- the failure is not in Python or CLaMP itself
- a symlink in `piano-clamp-env/bin/mscore` is not sufficient if the target
  binary cannot run on the compute node
- the current MuseScore 4 AppImage is too new for nodes limited to `glibc 2.28`

Recommended debugging sequence:

```bash
sacct -j <jobid> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End
cat /home/student/w/wshelor/share/piano-clamp-runs/slurm/mozart-passages-review-<jobid>.err
cat /home/student/w/wshelor/share/piano-clamp-runs/slurm/mozart-passages-review-<jobid>.out
```

If the error is `mscore` not found, resubmit with an explicit export:

```bash
sbatch --export=ALL,MUSESCORE_BIN="$HOME/miniforge3/envs/piano-clamp-env/bin/mscore" \
  /home/student/w/wshelor/share/piano-clamp-runs/jobs/queue_mozart_passage_review.sbatch
```

If the error mentions `GLIBC_2.34 not found` or missing shared libraries such
as `libasound.so.2`, the AppImage is incompatible with the compute node and
should not be used as-is. In that case, the next things to try are:

- check whether the cluster provides MuseScore as a module or system package
- use an older MuseScore build that is compatible with `glibc 2.28`
- if only MusicXML slicing is needed, consider replacing the MuseScore render
  dependency in the review workflow with a cluster-compatible alternative

## macOS Laptop Usage

Long local runs can keep a Mac awake with `--caffeinate`:
```bash
python scripts/embed_text_prompts.py \
  --config configs/embedding_config.yaml \
  --caffeinate
```

or:

```bash
python scripts/run_all_embeddings.py \
  --config configs/embedding_config.yaml \
  --text --symbolic --passages \
  --caffeinate
```

With `--passages`, the runner launches score, performance-MIDI, and aligned-audio
passage bundles. Add `--passage-source-material score` or repeat that option to
limit the passage families for a run.

`--caffeinate` is ignored on non-macOS systems.

## Outputs

Live outputs normally look like this under the configured roots:

```text
embeddings/
  text/
  symbolic/
  audio/
  passages/
analysis/
  similarities/
  nearest_prompts/
  nearest_passages/
  reports/
  frozen_manifests/
logs/
embedding_store/
  runs/
  registry.csv
  registry.json
```

Each embedding bundle contains:

- a `.npy` matrix
- a CSV table with row metadata and status fields
- `metadata.json`

Metadata records include:

- run ID and timestamp
- host, Python, Torch, CUDA, device
- project and upstream commits
- checkpoint path and hash
- config and input hashes
- normalization and embedding dimension
- stage-specific metadata such as prompt bank info or passage methods
- current run log path

## Important Scripts

The current operational scripts worth knowing first are:

- `scripts/check_installation.py`
- `scripts/prepare_cpc_adapter.py`
- `scripts/embed_text_prompts.py`
- `scripts/embed_symbolic_scores.py`
- `scripts/embed_audio.py`
- `scripts/embed_passages.py`
- `scripts/export_passage_review.py`
- `scripts/build_embedding_index.py`
- `scripts/compute_similarities.py`
- `scripts/validate_embeddings.py`
- `scripts/backfill_embedding_store.py`
- `scripts/audit_study_readiness.py`
- `scripts/freeze_corpus_manifest.py`
- `scripts/run_all_embeddings.py`

There are also older scripts in the repo from earlier stages of development.
When in doubt, use the scripts listed above first.

## Known Constraints

- the repo does not download copyrighted audio
- `.mscx` and `.mscz` are not direct embedding inputs; normalize them first with `scripts/convert_musescore_scores.py`
- phrase/cadence passage generation depends on explicit annotations
- audio passages are never inferred without alignment coverage
- C2 and SAAS spaces must be analyzed separately
- CPU runs are supported but much slower than GPU runs

## Suggested Reading Order For A New Agent

If you are orienting yourself in the codebase, read in this order:

1. this README
2. [configs/embedding_config.yaml](configs/embedding_config.yaml)
3. [docs/cpc_adapter.md](docs/cpc_adapter.md)
4. `scripts/embed_passages.py`
5. `src/piano_clamp/data_loading.py`
6. `src/piano_clamp/passage_generation.py`
7. `src/piano_clamp/embedding_store.py`

That sequence gives the fastest accurate picture of repo goals, current data
shape, and the active pipeline.
