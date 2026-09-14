# HPC Git Bridge Workflow

This project is designed for a split workflow:

- local machines are for code design, config design, review, and lightweight analysis;
- the HPC is for dataset access, embedding generation, and expensive analysis;
- git is the bridge for code, experiment configs, and lightweight run reports;
- shared dataset storage is the source of truth for musical assets.

The goal is that a local coding agent can help design experiments and interpret
results without needing direct access to Slurm, large datasets, embeddings,
audio, MIDI, or scratch files.

## Roles

### Local Checkout

The local checkout is the working design environment. Use it to:

- edit Python code, tests, docs, prompts, and experiment configs;
- run local tests that do not require the full dataset;
- review report artifacts produced on the HPC;
- plan the next batch of experiments.

The local checkout should not need to store large datasets, embeddings,
checkpoints, rendered audio, or large intermediate analysis files.

### HPC Checkout

The HPC checkout is the execution environment. Use it to:

- pull code and configs from git;
- access the shared datasets directory;
- run Slurm jobs;
- create embeddings and other large ignored artifacts;
- write lightweight reports that can be committed back to git.

Do not make one-off experimental code edits directly on the HPC unless they are
committed and pushed like any other change.

### Shared Datasets

Dataset creation and source-asset rendering belong outside this repo. The shared
datasets area should own:

- raw source files;
- cleaned symbolic files;
- rendered audio;
- alignments;
- dataset manifests;
- dataset version metadata;
- rights and provenance records.

`piano-clamp` consumes dataset manifests and existing assets. It validates,
embeds, extracts features, analyzes, and produces review/result artifacts.

## Git Policy

Commit and push:

- source code;
- tests;
- docs;
- prompt banks;
- experiment configs;
- small manifest examples or schemas;
- lightweight run reports.

Do not commit:

- embeddings;
- checkpoints;
- source audio;
- source MIDI;
- MusicXML/MXL corpus files;
- rendered audio;
- large arrays;
- temporary files;
- Slurm runtime logs;
- scratch directories.

Heavy artifacts should stay under ignored roots such as `embeddings/`, `logs/`,
scratch paths, or HPC run directories. Lightweight reports should live under a
non-ignored report root so they can be committed.

## Report Artifacts

Every operational script that runs on the HPC should produce a lightweight
report artifact, even when it fails. Logs are for debugging; reports are for
review, handoff, and local analysis.

A typical report bundle should look like:

```text
reports/
  hpc/
    <experiment_id>/
      <run_id>/
        report.md
        summary.json
        process.md
        warnings.csv
        failures.csv
        tables/
        figures/
        provenance.json
```

Reports should be small enough to commit. They should condense expensive
results and large-file analysis into forms that local agents can inspect without
loading the original embeddings or datasets.

At minimum, a report should include:

- script name and exact command line;
- timestamp, hostname, working directory, and Slurm job ID when available;
- git commit and whether the checkout was dirty;
- config path and resolved important config values;
- dataset root, manifest path, dataset version, and manifest hash when known;
- model/checkpoint paths and hashes when relevant;
- stage-by-stage status, counts, elapsed times, warnings, and failures;
- skipped records and reasons;
- paths and hashes for heavy artifacts left on the HPC;
- compact tables or summaries of computed results;
- enough diagnostic detail to debug common failures after the job is done.

For embedding and analysis jobs, reports should also include expensive summary
work that would be painful or impossible locally, such as:

- embedding row counts, dimensions, norm diagnostics, and duplicate checks;
- nearest-neighbor or top-k similarity tables;
- within-group and between-group similarity summaries;
- prompt extremes and outliers;
- feature-to-embedding correlations;
- cluster or projection summaries;
- compact figures suitable for review;
- caveats and suspicious patterns.

## Standard Cycle

### 1. Design Locally

On the local machine:

```bash
git pull
```

Edit code, docs, prompts, and experiment configs. Run local tests that do not
require the full dataset. Commit and push the changes:

```bash
git status
git add <files>
git commit -m "Describe the experiment or workflow change"
git push
```

### 2. Pull on the HPC

On the HPC:

```bash
cd /path/to/piano-clamp
git pull
```

Activate the project environment and confirm the expected dataset/run roots are
available. Machine-specific paths should come from ignored local config files or
environment variables, not from committed edits.

### 3. Validate Before Expensive Runs

Before launching long jobs, run the validation job or an equivalent short smoke
test. The validation job should check the environment, Python packages, CUDA
visibility, config resolution, and a small pipeline sample.

Use Slurm status tools to check progress:

```bash
squeue -u "$USER"
sacct -j <jobid> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End
```

Remember that `squeue` only shows pending and running jobs. Finished or quickly
failed jobs usually require `sacct` plus the Slurm output file.

### 4. Run the Experiment

Submit the Slurm job from the HPC checkout. Prefer parameterized jobs that take
the config, dataset root, run root, and report root from environment variables
or command-line arguments.

The job should print a final compact block like:

```text
PIANO_CLAMP_RUN_COMPLETE
status=ok
job_id=<slurm_job_id>
commit=<git_commit>
report_dir=reports/hpc/<experiment_id>/<run_id>
summary=reports/hpc/<experiment_id>/<run_id>/summary.json
heavy_artifacts=/path/on/hpc/to/ignored/artifacts
```

For failures, the job should still write a report and print:

```text
PIANO_CLAMP_RUN_COMPLETE
status=failed
job_id=<slurm_job_id>
commit=<git_commit>
report_dir=reports/hpc/<experiment_id>/<run_id>
summary=reports/hpc/<experiment_id>/<run_id>/summary.json
failure_stage=<stage_name>
```

### 5. Commit Reports From the HPC

After the run completes, review the report bundle. If it is lightweight and does
not contain large arrays, source data, embeddings, audio, or private scratch
files, commit and push it:

```bash
git status
git add reports/hpc/<experiment_id>/<run_id>
git commit -m "Add HPC report for <experiment_id> <run_id>"
git push
```

If a report is too large, reduce it before committing. The report should contain
derived summaries and compact tables, not raw embeddings or large source data.

### 6. Pull Reports Locally

Back on the local machine:

```bash
git pull
```

Local agents can now inspect the committed reports, review compact results, and
plan the next iteration without needing direct HPC access.

## Path Conventions

Committed configs should be portable. Machine-specific paths belong in ignored
files such as `configs/paths.yaml` or in environment variables.

Recommended environment concepts:

- dataset root: the shared parent directory containing dataset versions;
- run root: the ignored location for heavy outputs from `piano-clamp`;
- report root: the committed location for lightweight reports.

Avoid hardcoding a particular cluster path in reusable scripts. Cluster-specific
defaults may appear in personal wrapper scripts or ignored config files.

## Script Design Rules

Operational scripts should:

- accept explicit config paths;
- create a report bundle by default;
- write a report even on failure;
- separate heavy outputs from lightweight reports;
- record enough provenance to reproduce the run;
- include verbose process details in the report, not only in runtime logs;
- summarize large artifacts into compact tables and figures;
- make missing datasets, checkpoints, and environment problems fail clearly.

Scripts should not:

- create canonical datasets;
- render corpus-scale audio from MIDI or MusicXML;
- silently write heavy files into committed directories;
- require local machines to load large embeddings for basic review.

## Handoff To Local Agents

When asking a local coding agent to analyze a run, provide the committed report
path, not the raw HPC log. Good inputs are:

- `reports/hpc/<experiment_id>/<run_id>/report.md`;
- `reports/hpc/<experiment_id>/<run_id>/summary.json`;
- compact CSV tables under the report bundle;
- small figures under the report bundle.

If deeper debugging is needed, include the relevant Slurm output excerpt or the
path to the heavy artifact on the HPC, but keep the normal workflow centered on
committed reports.
