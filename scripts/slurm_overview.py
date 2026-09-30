#!/usr/bin/env python3
"""Quick overview of recent Slurm jobs for the current user.

Combines ``squeue`` (active jobs) and ``sacct`` (recent completed jobs) into
a concise human-readable table.  Suitable for running from the repo root:

    python scripts/slurm_overview.py --days 7 --limit 20

Notes:
- ``squeue`` only shows pending/running jobs; ``sacct`` is needed for
  completed/failed jobs (see ``logs/README.md``).
- Most repo Slurm logs use ``#SBATCH --output=logs/slurm-%x-%j.out``.
- Log lookup is configurable via ``--logs-dir`` so HPC runs using
  absolute/shared log paths can be inspected too.
- This is read-only; it does not mutate HPC artifacts.
"""

from __future__ import annotations

import argparse
import getpass
import glob as globmod
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SACCT_FIELDS = "JobIDRaw,JobID,JobName,State,ExitCode,Elapsed,Start,End,NodeList,ReqMem,MaxRSS,AllocTRES"
# squeue format: JobID|JobName|State|Elapsed|Start|End|NodeList|Partition|Reason|ReqMem|TimeLimit
SQUEUE_FORMAT = "%i|%j|%T|%M|%S|%e|%N|%P|%r|%m|%l"

FAIL_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "DEADLINE", "BOOT_FAIL"}

LOG_ERROR_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"traceback",
        r"error",
        r"exception",
        r"killed",
        r"oom",
        r"out of memory",
        r"command failed",
        r"no such file",
        r"permission denied",
        r"module not found",
        r"no module",
        r"cannot open",
        r"not found",
        r"slurmstepd.*error",
        r"glibc.*not found",
        r"libasound",
        r"mscore",
    ]
]

# Short pattern set used for WHY summarisation (user-facing concise reason)
SHORT_PATTERNS = [
    ("traceback", "traceback"),
    ("out of memory", "OOM"),
    ("oom", "OOM"),
    ("killed", "killed"),
    ("permission denied", "permission denied"),
    ("no such file", "no such file"),
    ("module not found", "module not found"),
    ("command failed", "command failed"),
    ("glibc", "glibc mismatch"),
    ("libasound", "missing libasound"),
    ("mscore", "mscore missing"),
    ("exception", "exception"),
    ("error", "error"),
]


@dataclass
class JobRow:
    raw: dict[str, str]  # original sacct/squeue row dict


@dataclass
class JobOverview:
    job_id: str
    job_name: str
    state: str
    exit_code: str
    elapsed: str
    start: str
    end: str
    nodelist: str
    req_mem: str
    max_rss: str
    alloc_tres: str
    source: str  # squeue|sacct|both
    children: list[dict[str, str]] = field(default_factory=list)
    log_path: str = ""
    log_snippet: list[str] = field(default_factory=list)
    failure_reason: str = ""
    is_active: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cmd: list[str], timeout: int = 15) -> tuple[int, str, str, str | None]:
    """Run *cmd*, return (returncode, stdout, stderr, error_message).

    If the binary is missing, error_message is set and returncode is 127.
    """
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr, None
    except FileNotFoundError:
        return 127, "", "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", "", f"command timed out: {' '.join(cmd)}"
    except Exception as exc:  # pragma: no cover - defensive
        return 1, "", "", str(exc)


def _parse_delimited(output: str, fields: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        # sacct --parsable2 and squeue -o with | may produce trailing |
        parts = line.split("|")
        # Pad or truncate to field count
        if len(parts) < len(fields):
            parts += [""] * (len(fields) - len(parts))
        elif len(parts) > len(fields):
            parts = parts[: len(fields)]
        row = {field: part.strip() for field, part in zip(fields, parts)}
        rows.append(row)
    return rows


def _base_job_id(job_id_raw: str) -> str:
    # sacct: 12345, 12345.batch, 12345.extern, 12345.0 etc
    # squeue: 12345
    if not job_id_raw:
        return ""
    return job_id_raw.split(".")[0].split("_")[0]


def _group_sacct_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        base = _base_job_id(row.get("JobIDRaw") or row.get("JobID") or "")
        if not base:
            continue
        grouped.setdefault(base, []).append(row)
    return grouped


def _pick_parent(rows: list[dict[str, str]]) -> dict[str, str]:
    """Pick the submitted-job row (no dot/batch/extern) if present."""
    for row in rows:
        jid_raw = row.get("JobIDRaw", "")
        if jid_raw and "." not in jid_raw and "_" not in jid_raw:
            return row
    for row in rows:
        jid = row.get("JobID", "")
        if jid and "." not in jid and "_" not in jid:
            return row
    # Fallback: shortest JobIDRaw
    return sorted(rows, key=lambda r: len(r.get("JobIDRaw", "")))[0]


def _detect_failure_state(state: str, exit_code: str) -> bool:
    state_upper = state.strip().upper()
    # Strip possible suffixes like "FAILED  " or "COMPLETED"
    base_state = state_upper.split()[0] if state_upper else ""
    if base_state in FAIL_STATES:
        return True
    if base_state != "COMPLETED" and base_state not in ("RUNNING", "PENDING", "COMPLETING", "CONFIGURING", "RESV_DEL_HOLD", "REQUEUE_FED", "REQUEUE_HOLD", "SUSPENDED"):
        # For sacct: COMPLETED is success, everything else non-COMPLETED that is not active is failure
        # But squeue active states are not failures - caller handles that
        pass
    # Non-zero exit code is failure regardless of state string
    exit_main = exit_code.split(":")[0].strip() if exit_code else "0"
    if exit_main not in ("", "0", "0:0"):
        # sacct ExitCode is like "1:0" - check first part
        try:
            if int(exit_main) != 0:
                return True
        except ValueError:
            return True
    if base_state == "FAILED":
        return True
    if base_state in FAIL_STATES:
        return True
    return False


def _summarise_exit(exit_code: str) -> str:
    if not exit_code or exit_code in ("0:0", "0"):
        return exit_code or ""
    return exit_code


def _failure_reason_for_group(
    parent: dict[str, str],
    children: list[dict[str, str]],
    log_snippet: list[str],
) -> str:
    state = (parent.get("State") or "").strip()
    state_upper = state.upper().split()[0] if state else ""
    exit_code = (parent.get("ExitCode") or "").strip()

    # Also check children for worse state
    worst_state = state_upper
    worst_exit = exit_code
    for child in children:
        cs = (child.get("State") or "").strip().upper().split()[0] if child.get("State") else ""
        ce = (child.get("ExitCode") or "").strip()
        # Prefer failure states
        if cs in FAIL_STATES and worst_state not in FAIL_STATES:
            worst_state = cs
            worst_exit = ce
        # Non-zero exit takes precedence
        ce_main = ce.split(":")[0] if ce else "0"
        we_main = worst_exit.split(":")[0] if worst_exit else "0"
        try:
            if int(ce_main) != 0 and int(we_main) == 0:
                worst_state = cs or worst_state
                worst_exit = ce
        except ValueError:
            pass

    # Determine reason string
    if worst_state == "TIMEOUT":
        return "TIMEOUT (walltime exceeded)"
    if worst_state == "OUT_OF_MEMORY":
        return "OUT_OF_MEMORY (OOM killed)"
    if worst_state == "CANCELLED":
        # Check if exit non-zero distinguishes user vs system cancel
        return f"CANCELLED (exit {worst_exit})" if worst_exit and worst_exit != "0:0" else "CANCELLED"
    if worst_state == "NODE_FAIL":
        return "NODE_FAIL"
    if worst_state == "DEADLINE":
        return "DEADLINE"
    if worst_state == "FAILED" or worst_state in FAIL_STATES:
        if worst_exit and worst_exit not in ("0:0", "0"):
            return f"{worst_state} (exit {worst_exit})"
        return worst_state
    # Non-zero exit without failure state
    if worst_exit and worst_exit not in ("0:0", "0", ""):
        try:
            main = int(worst_exit.split(":")[0])
            if main != 0:
                return f"FAILED (exit {worst_exit})"
        except ValueError:
            return f"FAILED (exit {worst_exit})"

    # If not flagged as failure yet but log hints at failure, surface it
    if log_snippet:
        # Use first matching short pattern for summary
        blob = " ".join(log_snippet).lower()
        for pat, label in SHORT_PATTERNS:
            if pat in blob:
                return f"error in log: {label}"
        return "error in log (see snippet)"

    # No clear failure
    if worst_state == "COMPLETED" and (not worst_exit or worst_exit in ("0:0", "0")):
        return ""
    if worst_state:
        return worst_state
    return ""


def _find_log_files(job_id: str, job_name: str, logs_dir: Path) -> list[Path]:
    """Find candidate log files for *job_id* in *logs_dir*.

    Patterns:
    - logs/slurm-<jobname>-<jobid>.out
    - logs/slurm-*<jobid>*.out
    - optionally .err too
    Returns existing paths, sorted with preferred first.
    """
    candidates: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        if p.is_file() and p.resolve() not in seen:
            candidates.append(p)
            seen.add(p.resolve())

    if not logs_dir.is_dir():
        return []

    # 1. Exact slurm-<jobname>-<jobid>.out/.err
    if job_name:
        for ext in (".out", ".err"):
            _add(logs_dir / f"slurm-{job_name}-{job_id}{ext}")

    # 2. Direct glob slurm-*<jobid>*.out/.err
    for ext in (".out", ".err"):
        pattern = str(logs_dir / f"slurm-*{job_id}*{ext}")
        for match in globmod.glob(pattern):
            _add(Path(match))

    # 3. Any *<jobid>*.out/.err
    for ext in (".out", ".err"):
        pattern = str(logs_dir / f"*{job_id}*{ext}")
        for match in globmod.glob(pattern):
            _add(Path(match))

    return candidates


def _extract_log_snippet(log_path: Path, max_lines: int = 8, tail_bytes: int = 200_000) -> list[str]:
    """Return up to *max_lines* lines from the log that match error patterns.

    For performance, reads only the last *tail_bytes* bytes if file is large.
    """
    if not log_path.is_file():
        return []
    try:
        size = log_path.stat().st_size
        if size > tail_bytes:
            with log_path.open("rb") as fh:
                fh.seek(max(0, size - tail_bytes))
                # Drop partial first line
                fh.readline()
                text = fh.read().decode("utf-8", errors="replace")
        else:
            text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    lines = text.splitlines()
    matches: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        for pat in LOG_ERROR_PATTERNS:
            if pat.search(line):
                # Include line number context
                prefix = f"line {idx+1}: "
                # Truncate very long lines
                display = line.strip()
                if len(display) > 300:
                    display = display[:300] + "..."
                matches.append(prefix + display)
                break
        if len(matches) >= max_lines:
            break

    # If no pattern matched but file is non-empty, return last few non-empty lines for context
    # Only for files that look like they contain errors (check file size small) — skip if matches empty
    return matches


def _tail_file(log_path: Path, n: int = 50) -> list[str]:
    if not log_path.is_file():
        return []
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"(could not read log: {exc})"]
    lines = text.splitlines()
    # Return last n non-empty or not? Return last n regardless
    tail = lines[-n:]
    # Truncate long lines
    return [line[:400] for line in tail]


# ---------------------------------------------------------------------------
# Query Slurm
# ---------------------------------------------------------------------------


def _query_squeue(user: str) -> tuple[list[dict[str, str]], str | None]:
    fields = ["JobID", "JobName", "State", "Elapsed", "Start", "End", "NodeList", "Partition", "Reason", "ReqMem", "TimeLimit"]
    cmd = ["squeue", "-u", user, "-h", "-o", SQUEUE_FORMAT]
    rc, out, err, missing = _run(cmd)
    if missing:
        return [], missing
    if rc != 0:
        # squeue returns non-zero when no jobs? Some versions return 0 with empty.
        # Treat empty output as no active jobs, but surface stderr.
        if not out.strip():
            msg = err.strip() or f"squeue exited {rc}"
            # If stderr mentions invalid user / no jobs, treat as empty but warn
            if "no jobs" in msg.lower() or "invalid" in msg.lower():
                return [], None
            return [], msg
        return [], err.strip() or f"squeue exited {rc}"
    if not out.strip():
        return [], None
    rows = _parse_delimited(out, fields)
    # Enrich with ExitCode placeholder and map to sacct-like fields
    for row in rows:
        row["JobIDRaw"] = row.get("JobID", "")
        row["JobID_display"] = row.get("JobID", "")
        row["ExitCode"] = ""
        row["MaxRSS"] = ""
        row["AllocTRES"] = ""
        row["ReqMem"] = row.get("ReqMem", "")
    return rows, None


def _query_sacct(user: str, days: int, job_filter: str | None = None) -> tuple[list[dict[str, str]], str | None]:
    fields = [f.strip() for f in SACCT_FIELDS.split(",")]
    # sacct expects --starttime format; use now-Ndays
    starttime = f"now-{days}days"
    cmd = ["sacct", "--parsable2", "--noheader", "--allusers" if False else "--user", user, "--starttime", starttime, "--format", SACCT_FIELDS]
    if job_filter:
        # When inspecting single job, sacct -j <id> is more reliable than date filter
        cmd = ["sacct", "--parsable2", "--noheader", "-j", job_filter, "--format", SACCT_FIELDS]
    rc, out, err, missing = _run(cmd)
    if missing:
        return [], missing
    if rc != 0:
        # sacct may fail if no accounting data yet or slurm not configured
        if not out.strip():
            return [], err.strip() or f"sacct exited {rc}"
        # Still try to parse partial output
    if not out.strip():
        return [], None
    rows = _parse_delimited(out, fields)
    return rows, None


# ---------------------------------------------------------------------------
# Build overview
# ---------------------------------------------------------------------------


def _build_overviews(
    sacct_grouped: dict[str, list[dict[str, str]]],
    squeue_rows: list[dict[str, str]],
    logs_dir: Path,
    fetch_snippets: bool = True,
) -> list[JobOverview]:
    overviews: dict[str, JobOverview] = {}

    # First, from sacct grouped
    for base_id, rows in sacct_grouped.items():
        parent = _pick_parent(rows)
        children = [r for r in rows if r is not parent]
        job_name = (parent.get("JobName") or "").strip()
        state = (parent.get("State") or "").strip()
        exit_code = (parent.get("ExitCode") or "").strip()
        # For sacct, parent state already reflects final; but verify children for worst
        log_paths = _find_log_files(base_id, job_name, logs_dir) if fetch_snippets else []
        log_path_str = str(log_paths[0]) if log_paths else ""
        snippet: list[str] = []
        if log_paths and fetch_snippets:
            # Check failure-ish first; but we fetch snippet regardless for failed jobs to determine reason
            # Quick check: is this job failed?
            is_failed_like = _detect_failure_state(state, exit_code)
            # Also peek children
            for ch in children:
                if _detect_failure_state(ch.get("State", ""), ch.get("ExitCode", "")):
                    is_failed_like = True
                    break
            if is_failed_like:
                snippet = _extract_log_snippet(log_paths[0])

        # Build failure reason (needs snippet)
        tmp_parent_for_reason = parent
        reason = _failure_reason_for_group(tmp_parent_for_reason, children, snippet)
        # But if reason empty and we detected failure but no snippet, still want non-empty
        # _failure_reason handles that.

        ov = JobOverview(
            job_id=base_id,
            job_name=job_name,
            state=state,
            exit_code=exit_code,
            elapsed=(parent.get("Elapsed") or "").strip(),
            start=(parent.get("Start") or "").strip(),
            end=(parent.get("End") or "").strip(),
            nodelist=(parent.get("NodeList") or "").strip(),
            req_mem=(parent.get("ReqMem") or "").strip(),
            max_rss=(parent.get("MaxRSS") or "").strip(),
            alloc_tres=(parent.get("AllocTRES") or "").strip(),
            source="sacct",
            children=children,
            log_path=log_path_str,
            log_snippet=snippet,
            failure_reason=reason,
            is_active=False,
        )
        overviews[base_id] = ov

    # Merge squeue active jobs
    for row in squeue_rows:
        jid = (row.get("JobID") or "").strip()
        if not jid:
            continue
        base = _base_job_id(jid)
        if base in overviews:
            # Mark as active (squeue shows it's still active, sacct may already have it but state pending)
            overviews[base].is_active = True
            overviews[base].source = "both"
            # Prefer squeue's state for active jobs (more current)
            sq_state = (row.get("State") or "").strip()
            if sq_state:
                overviews[base].state = sq_state
                # Active jobs are not failed; clear failure reason if it was stale
                if sq_state.upper() in ("RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "SUSPENDED"):
                    overviews[base].failure_reason = ""
                    overviews[base].is_active = True
        else:
            # Active job not yet in sacct
            job_name = (row.get("JobName") or "").strip()
            log_paths = _find_log_files(base, job_name, logs_dir) if fetch_snippets else []
            log_path_str = str(log_paths[0]) if log_paths else ""
            ov = JobOverview(
                job_id=base,
                job_name=job_name,
                state=(row.get("State") or "").strip(),
                exit_code="",
                elapsed=(row.get("Elapsed") or "").strip(),
                start=(row.get("Start") or "").strip(),
                end=(row.get("End") or "").strip(),
                nodelist=(row.get("NodeList") or "").strip(),
                req_mem=(row.get("ReqMem") or "").strip(),
                max_rss="",
                alloc_tres="",
                source="squeue",
                children=[],
                log_path=log_path_str,
                log_snippet=[],
                failure_reason="",
                is_active=True,
            )
            overviews[base] = ov

    # Sort by start/end descending where possible, else job_id numeric descending
    def _sort_key(ov: JobOverview) -> Any:
        # Try parse start as datetime string from sacct: YYYY-MM-DDTHH:MM:SS
        for ts in (ov.start, ov.end):
            if ts and ts not in ("Unknown", "None", ""):
                try:
                    # sacct format is YYYY-MM-DDTHH:MM:SS
                    dt = datetime.fromisoformat(ts)
                    return dt
                except ValueError:
                    try:
                        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
                        return dt
                    except ValueError:
                        pass
        # Fallback sortable by job id numeric
        try:
            return datetime.min
        except Exception:
            return datetime.min

    # Actually sort by job_id numeric descending as tie-breaker
    sorted_ovs = sorted(
        overviews.values(),
        key=lambda ov: (_sort_key(ov), int(ov.job_id) if ov.job_id.isdigit() else ov.job_id),
        reverse=True,
    )
    return sorted_ovs


def _status_label(ov: JobOverview) -> str:
    if ov.is_active:
        return ov.state or "ACTIVE"
    state_upper = (ov.state or "").upper().split()[0] if ov.state else ""
    if state_upper == "COMPLETED" and (not ov.exit_code or ov.exit_code in ("0:0", "0")):
        return "COMPLETED"
    if ov.failure_reason:
        return state_upper or "FAILED"
    if state_upper == "COMPLETED":
        return "COMPLETED"
    return state_upper or "UNKNOWN"


def _success_failed_active_counts(overviews: list[JobOverview]) -> tuple[int, int, int]:
    active = sum(1 for ov in overviews if ov.is_active)
    failed = sum(1 for ov in overviews if not ov.is_active and ov.failure_reason)
    succeeded = sum(
        1
        for ov in overviews
        if not ov.is_active and not ov.failure_reason and (ov.state or "").upper().startswith("COMPLETED")
    )
    return succeeded, failed, active


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _print_table(overviews: list[JobOverview], limit: int, warnings: list[str]) -> None:
    if warnings:
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)

    if not overviews:
        print("No Slurm jobs found for the given user/time window.")
        print("Hints: check --user, --days, and whether sacct/squeue are available on this host.")
        return

    truncated = overviews[:limit]
    succeeded, failed, active = _success_failed_active_counts(overviews)

    # Header
    print(f"Recent Slurm jobs (showing {len(truncated)} of {len(overviews)}; succeeded={succeeded} failed={failed} active={active})")
    print("")

    # Compute column widths dynamically but capped
    headers = ["JOBID", "NAME", "STATE", "EXIT", "ELAPSED", "START", "END", "WHY"]
    # Build rows for width calc
    table_rows: list[list[str]] = []
    for ov in truncated:
        why = ov.failure_reason or ("running" if ov.is_active else ("succeeded" if (ov.state or "").upper().startswith("COMPLETED") else ""))
        # Truncate WHY for table
        why_short = why[:50] + "..." if len(why) > 50 else why
        table_rows.append(
            [
                ov.job_id,
                ov.job_name[:30],
                _status_label(ov)[:14],
                ov.exit_code or ("-" if ov.is_active else ov.exit_code),
                ov.elapsed,
                ov.start.replace("T", " ")[:19] if ov.start not in ("Unknown", "") else ov.start,
                ov.end.replace("T", " ")[:19] if ov.end not in ("Unknown", "") else ov.end,
                why_short,
            ]
        )

    # Fixed widths - simple formatted table
    col_widths = [8, 22, 12, 8, 10, 19, 19, 30]
    # Adjust JOBID width to max
    for i, h in enumerate(headers):
        max_len = max(len(h), max((len(r[i]) for r in table_rows), default=0))
        col_widths[i] = min(max_len + 2, col_widths[i] if i != 7 else 50)
    # Ensure minimal
    col_widths[0] = max(col_widths[0], 8)
    col_widths[1] = max(col_widths[1], 10)

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * min(len(h), w) for h, w in zip(headers, col_widths)]))
    for row in table_rows:
        # Truncate each cell to width
        truncated_row = [cell[:w] for cell, w in zip(row, col_widths)]
        print(fmt.format(*truncated_row))

    # Failure details section
    failed_jobs = [ov for ov in truncated if ov.failure_reason]
    if failed_jobs:
        print("")
        print("Failures:")
        for ov in failed_jobs:
            print(f"  {ov.job_id} ({ov.job_name}) {ov.state} exit={ov.exit_code or 'n/a'} -> {ov.failure_reason}")
            if ov.log_path:
                print(f"    log: {ov.log_path}")
                if ov.log_snippet:
                    for line in ov.log_snippet[:5]:
                        print(f"      {line}")
                else:
                    print(f"      (no error pattern matched in log; check full log)")
            else:
                print(f"    log: not found in logs dir (tried logs/slurm-{ov.job_name}-{ov.job_id}.out and logs/slurm-*{ov.job_id}*.out/.err)")
                # Still hint at log snippet if we have one from alternate path? already empty

    # Active jobs section
    active_jobs = [ov for ov in truncated if ov.is_active]
    if active_jobs:
        print("")
        print("Active (squeue):")
        for ov in active_jobs:
            print(f"  {ov.job_id} {ov.job_name} {ov.state} elapsed={ov.elapsed} start={ov.start}")


def _overviews_to_json(overviews: list[JobOverview], warnings: list[str], limit: int) -> dict[str, Any]:
    truncated = overviews[:limit]
    succeeded, failed, active = _success_failed_active_counts(overviews)
    jobs = []
    for ov in truncated:
        jobs.append(
            {
                "job_id": ov.job_id,
                "job_name": ov.job_name,
                "state": ov.state,
                "exit_code": ov.exit_code,
                "elapsed": ov.elapsed,
                "start": ov.start,
                "end": ov.end,
                "nodelist": ov.nodelist,
                "req_mem": ov.req_mem,
                "max_rss": ov.max_rss,
                "alloc_tres": ov.alloc_tres,
                "source": ov.source,
                "is_active": ov.is_active,
                "failure_reason": ov.failure_reason,
                "success": not ov.is_active and not ov.failure_reason and (ov.state or "").upper().startswith("COMPLETED"),
                "log_path": ov.log_path,
                "log_snippet": ov.log_snippet,
            }
        )
    return {
        "total": len(overviews),
        "shown": len(truncated),
        "succeeded": succeeded,
        "failed": failed,
        "active": active,
        "warnings": warnings,
        "jobs": jobs,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    default_user = os.environ.get("USER") or os.environ.get("LOGNAME") or getpass.getuser()
    parser = argparse.ArgumentParser(
        description="Summarize recent Slurm jobs for the current user (squeue + sacct).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/slurm_overview.py --days 7 --limit 20\n"
            "  python scripts/slurm_overview.py --days 14 --limit 25\n"
            "  python scripts/slurm_overview.py --job 123456 --logs-dir logs\n"
            "  python scripts/slurm_overview.py --json --days 1 | jq .\n"
        ),
    )
    parser.add_argument("--days", type=int, default=7, help="Look back N days in sacct (default: 7)")
    parser.add_argument("--limit", type=int, default=20, help="Max jobs to show (default: 20)")
    parser.add_argument("--user", type=str, default=default_user, help="Slurm user to query (default: $USER)")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory to search for Slurm log files (default: logs)")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--job", type=str, default=None, help="Inspect a single JOBID deeply (shows all sacct rows and log tail)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.days < 1:
        parser.error("--days must be >= 1")
    if args.limit < 1:
        parser.error("--limit must be >= 1")

    logs_dir = Path(args.logs_dir)
    # If relative, it is relative to cwd (repo root when run as instructed)
    # Keep as-is; find helpers use is_dir check.

    warnings: list[str] = []

    # Single-job deep dive
    if args.job:
        job_id = args.job.strip()
        # Query sacct for that job (no days filter)
        sacct_rows, sacct_warn = _query_sacct(args.user, args.days, job_filter=job_id)
        if sacct_warn:
            if "command not found" in sacct_warn:
                warnings.append(f"sacct unavailable: {sacct_warn}")
            else:
                warnings.append(f"sacct: {sacct_warn}")

        squeue_rows, squeue_warn = _query_squeue(args.user)
        if squeue_warn and "command not found" in squeue_warn:
            warnings.append(f"squeue unavailable: {squeue_warn}")
        elif squeue_warn:
            warnings.append(f"squeue: {squeue_warn}")

        # Filter squeue to this job only for display
        squeue_rows = [r for r in squeue_rows if _base_job_id(r.get("JobID", "")) == job_id]

        if not sacct_rows and not squeue_rows:
            if args.json:
                payload: dict[str, Any] = {
                    "job_id": job_id,
                    "warnings": warnings,
                    "sacct_rows": [],
                    "squeue_rows": [],
                    "log_path": "",
                    "log_tail": [],
                }
                print(json.dumps(payload, indent=2))
                return 0
            if warnings:
                for w in warnings:
                    print(f"warning: {w}", file=sys.stderr)
            print(f"No sacct/squeue data found for job {job_id}.")
            print(f"Checked: sacct -j {job_id} and squeue -u {args.user}")
            if sacct_warn and "command not found" in sacct_warn:
                print("Hint: Slurm does not appear to be installed on this host.")
            # Still try log lookup
            # Need job_name guess - try to find any log matching job_id
            log_paths = _find_log_files(job_id, "", logs_dir)
            if log_paths:
                print(f"Found log(s) for {job_id}:")
                for lp in log_paths:
                    print(f"  {lp}")
                    tail = _tail_file(lp, 50)
                    for line in tail[-20:]:
                        print(f"    {line}")
            else:
                print(f"No log found matching *{job_id}* in {logs_dir}/")
                print(f"Hint: try --logs-dir /share/users/.../logs or absolute Slurm log path")
            return 0

        # Group for failure reason even in single-job mode
        grouped = _group_sacct_rows(sacct_rows)
        # For single job, find its group
        group_rows = grouped.get(job_id, sacct_rows)

        # Log lookup - need job name from parent
        job_name = ""
        if group_rows:
            parent = _pick_parent(group_rows)
            job_name = (parent.get("JobName") or "").strip()
        else:
            # From squeue
            if squeue_rows:
                job_name = (squeue_rows[0].get("JobName") or "").strip()

        log_paths = _find_log_files(job_id, job_name, logs_dir)
        log_path_str = str(log_paths[0]) if log_paths else ""

        # Build JSON or human output
        if args.json:
            payload = {
                "job_id": job_id,
                "warnings": warnings,
                "sacct_rows": sacct_rows,
                "squeue_rows": squeue_rows,
                "log_path": log_path_str,
                "log_paths": [str(p) for p in log_paths],
                "log_snippet": _extract_log_snippet(log_paths[0]) if log_paths else [],
                "log_tail": _tail_file(log_paths[0], 80) if log_paths else [],
            }
            print(json.dumps(payload, indent=2))
            return 0

        if warnings:
            for w in warnings:
                print(f"warning: {w}", file=sys.stderr)

        print(f"Job {job_id} details")
        print(f"  logs dir: {logs_dir} ({'exists' if logs_dir.is_dir() else 'not found'})")
        print("")
        if sacct_rows:
            print(f"sacct -j {job_id} rows ({len(sacct_rows)}):")
            hdr = SACCT_FIELDS.split(",")
            print("  " + " | ".join(hdr))
            for row in sacct_rows:
                print("  " + " | ".join(row.get(h, "") for h in hdr))
            print("")
            if grouped.get(job_id):
                parent = _pick_parent(grouped[job_id])
                snippet = _extract_log_snippet(log_paths[0]) if log_paths else []
                reason = _failure_reason_for_group(parent, [r for r in grouped[job_id] if r is not parent], snippet)
                if reason:
                    print(f"  failure reason: {reason}")
                else:
                    state = (parent.get("State") or "").upper()
                    exitc = (parent.get("ExitCode") or "")
                    if state.startswith("COMPLETED") and exitc in ("0:0", "0", ""):
                        print(f"  status: succeeded (COMPLETED, exit {exitc or '0:0'})")
                    else:
                        print(f"  state: {parent.get('State')} exit={parent.get('ExitCode')} elapsed={parent.get('Elapsed')}")
                print("")
        if squeue_rows:
            print(f"squeue (active) for {job_id}:")
            for row in squeue_rows:
                print(f"  {row.get('JobID')} {row.get('JobName')} {row.get('State')} elapsed={row.get('Elapsed')} start={row.get('Start')} end={row.get('End')} nodelist={row.get('NodeList')}")
            print("")

        if log_paths:
            for lp in log_paths[:2]:
                print(f"log: {lp}")
                snippet = _extract_log_snippet(lp)
                if snippet:
                    print("  error snippet:")
                    for line in snippet[:8]:
                        print(f"    {line}")
                tail = _tail_file(lp, 50)
                if tail:
                    print(f"  tail (last {min(30, len(tail))} lines):")
                    for line in tail[-30:]:
                        print(f"    {line}")
                print("")
        else:
            print(f"log: not found in {logs_dir}/ (tried slurm-{job_name}-{job_id}.out, slurm-*{job_id}*.out/.err, *{job_id}*.out/.err)")
            print(f"Hint: if HPC runs use absolute/shared log paths, re-run with --logs-dir /absolute/path/to/logs")
        return 0

    # Normal overview mode
    sacct_rows, sacct_warn = _query_sacct(args.user, args.days)
    if sacct_warn:
        if "command not found" in sacct_warn:
            warnings.append(f"sacct unavailable: {sacct_warn} (install Slurm or run on HPC)")
        else:
            warnings.append(f"sacct: {sacct_warn}")

    squeue_rows, squeue_warn = _query_squeue(args.user)
    if squeue_warn:
        if "command not found" in squeue_warn:
            warnings.append(f"squeue unavailable: {squeue_warn} (install Slurm or run on HPC)")
        else:
            warnings.append(f"squeue: {squeue_warn}")

    grouped = _group_sacct_rows(sacct_rows) if sacct_rows else {}
    overviews = _build_overviews(grouped, squeue_rows, logs_dir, fetch_snippets=True)

    if args.json:
        payload = _overviews_to_json(overviews, warnings, args.limit)
        # Include query metadata
        payload["query"] = {"user": args.user, "days": args.days, "limit": args.limit, "logs_dir": str(logs_dir)}
        print(json.dumps(payload, indent=2))
        return 0

    _print_table(overviews, args.limit, warnings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
