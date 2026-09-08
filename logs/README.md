# Runtime logs

This directory is the default log root for:

- stage logs written by embedding scripts
- Slurm stdout/stderr files
- upstream CLaMP diagnostics copied into the run logs

Current stage logging is per-run rather than append-forever. The corresponding
`metadata.json` for an embedding bundle records the exact `log_path` used for
that run.

These files are generated artifacts and are ignored by git.

Operational note for passage review jobs:

- `squeue` only shows pending/running jobs; a fast failure will disappear immediately
- use `sacct -j <jobid> ...` to confirm the final state
- inspect the corresponding `mozart-passages-review-<jobid>.err` file for the real cause
- on August 26, 2026, the key failure signatures were `mscore` not found, then
  `GLIBC_2.34 not found` and missing `libasound.so.2` from a MuseScore 4 AppImage
