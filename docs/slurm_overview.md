# Slurm Overview (`scripts/slurm_overview.py`)

Quick, read-only overview of recent Slurm jobs for the current user. Run from the repo root:

```bash
python scripts/slurm_overview.py --days 7 --limit 20
```

## What it does

Combines three data sources into one concise table / JSON:

| Source | Command | Purpose |
|--------|---------|---------|
| **Active jobs** | `squeue -u USER` | Live scheduler state: `RUNNING`, `PENDING`, etc. |
| **Historical/final state** | `sacct --starttime now-Ndays --user USER --format JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End,NodeList,ReqMem,MaxRSS,AllocTRES` | Final state, exit code, timings from accounting DB. |
| **Log files** | `logs/slurm-<jobname>-<jobid>.out` (and globs `slurm-*<jobid>*.out/.err`, `*<jobid>*.out/.err`) under `--logs-dir` (default `logs`) | Best-effort failure clues via pattern matching (traceback, error, oom, etc.). |

Child rows like `12345.batch` / `12345.extern` are collapsed into the parent `12345` summary, but the parent's `State`/`ExitCode` is checked against children to surface the worst failure.

## Usage

```bash
# 1) Overview of recent jobs (last 14 days, 25 rows)
python scripts/slurm_overview.py --days 14 --limit 25

# 2) Inspect one job deeply (all sacct rows + log tail/snippet)
python scripts/slurm_overview.py --job 123456

# 3) JSON output for automation
python scripts/slurm_overview.py --json --days 1 | jq .
python scripts/slurm_overview.py --job 123456 --json | jq .

# 4) Using a non-default / shared HPC log directory
python scripts/slurm_overview.py --logs-dir /share/users/$USER/piano-clamp/logs
python scripts/slurm_overview.py --job 123456 --logs-dir /home/student/w/wshelor/share/piano-clamp-runs/slurm
```

`--help` documents all columns, fields, states, and limitations:

```bash
python scripts/slurm_overview.py --help
```

## Sample human-readable output

Default command produces a compact table. `END` is trimmed to `YYYY-MM-DDTHH:MM`; `REASON` is best-effort (see below).

```
Recent Slurm jobs (showing 3 of 3; succeeded=1 failed=1 active=1)

JOBID      NAME                 STATE       STATUS     EXIT  ELAPSED   END               REASON
123456     embed-all            COMPLETED   success    0:0   01:12:44  2026-09-30T04:12    -
123457     embed-bach-symbolic  FAILED      failed     1:0   00:03:21  2026-09-30T04:20    ModuleNotFoundError: No module named torch in logs/slurm-embed-bach-symbolic-123457.out
123458     validate-hpc         RUNNING     active     -     00:18:03  -                 currently running

Failures:
  123457 (embed-bach-symbolic) FAILED exit=1:0 -> FAILED (exit 1:0)
    log: logs/slurm-embed-bach-symbolic-123457.out
      line 42: ModuleNotFoundError: No module named torch

Active (squeue):
  123458 validate-hpc RUNNING elapsed=00:18:03 start=2026-09-30T04:02:00
```

*Verify:* `STATE` is raw Slurm, `STATUS` is derived `success|failed|active`, `EXIT` is `0:0` vs `1:0`, `ELAPSED`/`END` parsed, `REASON` contains error snippet + log path when failed.

## Sample JSON output (`--json`)

`--json` emits an object with `jobs` as a **list of stable objects** (suitable for `jq`). Each object has the same keys:

```json
[
  {
    "job_id": "123457",
    "job_name": "embed-bach-symbolic",
    "state": "FAILED",
    "status": "failed",
    "exit_code": "1:0",
    "elapsed": "00:03:21",
    "start": "2026-09-30T04:17:00",
    "end": "2026-09-30T04:20:21",
    "source": ["sacct", "log"],
    "log_path": "logs/slurm-embed-bach-symbolic-123457.out",
    "reason": "ModuleNotFoundError: No module named torch in logs/slurm-embed-bach-symbolic-123457.out",
    "details": {
      "node_list": "gpu001",
      "matched_log_lines": [
        "line 41: Traceback (most recent call last):",
        "line 42: ModuleNotFoundError: No module named torch"
      ]
    }
  }
]
```

Actual `--json` top-level is `{"total":..., "shown":..., "succeeded":..., "failed":..., "active":..., "warnings":..., "jobs": [...], "query": {...}}` where `jobs` is the list above. Single-job mode (`--job ID --json`) emits a single object with the same field names plus `sacct_rows`/`squeue_rows`/`log_tail` for deep dive.

## Output fields / columns

| Field (JSON) | Column (table) | Meaning |
|--------------|----------------|---------|
| `job_id` | `JOBID` | Numeric Slurm job id (base id without `.batch`/`.extern`). String in JSON for stability. |
| `job_name` | `NAME` | `--job-name` (`%x`) from sbatch. Truncated to 24 chars in table. |
| `state` | `STATE` | Raw Slurm state: `COMPLETED`, `FAILED`, `RUNNING`, `PENDING`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, `NODE_FAIL`, etc. Upper-cased first token from sacct/squeue. |
| `status` | `STATUS` | Derived summary: `success` (COMPLETED + 0:0), `failed` (FAILED or non-zero exit or TIMEOUT/OOM/NODE_FAIL), `active` (RUNNING/PENDING from squeue), `cancelled`/`timeout`/`oom`/`node_fail` for specific failures, `unknown` otherwise. |
| `exit_code` | `EXIT` | sacct `ExitCode` as `exit:signal` (e.g. `0:0`, `1:0`). `"-"` if still active. Non-zero `exit` ⇒ failure even if state is `COMPLETED`. |
| `elapsed` | `ELAPSED` | Wall time used (`HH:MM:SS`, from sacct `Elapsed` or squeue `TimeUsed`). |
| `start` | *(JSON only; table shows via END + details)* | Start timestamp `YYYY-MM-DDTHH:MM:SS` from sacct (`Start`) or squeue. Empty/`Unknown` if not started. Available in JSON and ` --job` details. |
| `end` | `END` | End timestamp `YYYY-MM-DDTHH:MM` (trimmed to minute). `"-"` if active or `Unknown`. |
| `source` | *(JSON only)* | List of data sources that contributed: `["sacct"]`, `["squeue"]`, `["sacct","squeue"]`, plus `"log"` if a log file was found. String `sacct|squeue|both` collapsed internally but exposed as list in JSON. |
| `log_path` | *(shown in table via REASON + Failures section)* | Best log match in `--logs-dir`: prefers `slurm-<name>-<id>.out`, then `slurm-*<id>*.out/.err`, then `*<id>*.out/.err`. Empty if not found. |
| `reason` | `REASON` | Best-effort human reason (state + exit + log). Examples: `"-"` (success), `"currently running"` (active), `"FAILED (exit 1:0)"`, `"TIMEOUT (walltime exceeded)"`, `"OUT_OF_MEMORY (OOM killed)"`, `"ModuleNotFoundError: ... in logs/..."`. See *Reason* below. |

`details` (JSON) also exposes `node_list` (`NodeList`), `matched_log_lines` (up to 8 error-matching lines with `line N:` prefix), `req_mem`, `max_rss`, `alloc_tres`, and raw `failure_reason`/`is_active` for debugging.

## How to interpret common states

| STATE | STATUS | Typical meaning |
|-------|--------|-----------------|
| `RUNNING` | `active` | Executing on a node (`squeue`). `EXIT="-"`, `END="-"`, `REASON="currently running"`. |
| `PENDING` | `active` | Queued, waiting for resources (`Reason` column from squeue, e.g. `Resources`, `Priority`). |
| `COMPLETED` | `success` | Finished with `EXIT=0:0`. `REASON="-"`. |
| `FAILED` | `failed` | Step or script returned non-zero `EXIT` (e.g. `1:0`). Check `REASON` + log snippet (often `Traceback`, `ModuleNotFoundError`, `No such file`, `Permission denied`). |
| `CANCELLED` | `failed` (`cancelled`) | Killed by user `scancel` or dependency. `EXIT` may be `0:0` or non-zero. |
| `TIMEOUT` | `failed` (`timeout`) | Exceeded `#SBATCH --time`. `REASON="TIMEOUT (walltime exceeded)"`. |
| `OUT_OF_MEMORY` | `failed` (`oom`) | Exceeded `--mem`/`--mem-per-cpu` or `MaxRSS ≈ ReqMem`. Logs may contain `oom`, `out of memory`, `killed`, `slurmstepd: error`. |
| `NODE_FAIL` | `failed` (`node_fail`) | Node failure; often requeueable. Check `NodeList` + cluster status. |

`BOOT_FAIL`, `DEADLINE`, `SUSPENDED`, `CONFIGURING` etc. map to `failed` or `active` via `status`; `State` is preserved verbatim for inspection.

## Reason (best-effort)

`reason` / `REASON` is derived in priority order:

1. Slurm `STATE` in `FAIL_STATES` (`TIMEOUT`, `OUT_OF_MEMORY`, `CANCELLED`, `NODE_FAIL`, `FAILED`, …) → e.g. `TIMEOUT (walltime exceeded)`.
2. Non-zero `ExitCode` (`1:0`, `2:0`, …) → `FAILED (exit 1:0)`.
3. Log snippet: first error-matching lines from the log's last ~200 KB (patterns: `traceback`, `error`, `exception`, `killed`, `oom`/`out of memory`, `command failed`, `no such file`, `permission denied`, `module not found`/`no module`, `cannot open`, `not found`, `slurmstepd`, `glibc.*not found`, `libasound`, `mscore`). Table shows `"<snippet> in <log_path>"` (truncated to 70 chars); JSON stores full `matched_log_lines`.
4. Otherwise `"-"` (success) or `"currently running"` (active).

Thus a parser is working if: `COMPLETED+0:0 → status=success, reason="-"`; `FAILED+1:0 + log containing ModuleNotFoundError → status=failed, reason contains ModuleNotFoundError and log_path`.

## Limitations

- **`squeue` only shows active jobs.** Completed/failed jobs disappear immediately; use `sacct` history. Script merges both so active jobs appear even if not yet in `sacct`.
- **`sacct` availability depends on cluster accounting.** Retention window, `SlurmDBD` config, and `--starttime` (`--days N`, default 7) limit what is returned. On non-HPC hosts `sacct`/`squeue` are missing → warnings, empty table, JSON with `warnings`, no crash.
- **Logs can be missing.** If jobs used a different `#SBATCH --output` (absolute/shared path like `/share/.../slurm-%x-%j.out`) and `--logs-dir` is not set to that directory, `log_path` will be empty and `reason` falls back to state/exit only.
- **Child rows are collapsed.** `sacct` emits parent + `.batch` + `.extern` rows; the overview groups by base job id and picks the parent row for `State/ExitCode/Elapsed/Start/End`, but checks children for the worst failure (so a `.batch` FAILED is not hidden). Per-step details remain visible via ` --job <id>` (shows all sacct rows).

## See also

- `logs/README.md` — why `squeue` vs `sacct` matters and where logs live.
- `README.md` (§ HPC Usage, ~353-420) — `sbatch` workflow and `#SBATCH --output=logs/slurm-%x-%j.out`.
- `jobs/*.slurm` — job templates.
