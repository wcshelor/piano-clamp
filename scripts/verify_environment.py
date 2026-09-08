#!/usr/bin/env python3
"""Perform read-only checks of the local CLaMP 3 research environment."""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from piano_clamp.embeddings import C2_CHECKPOINT, SAAS_CHECKPOINT, UPSTREAM_COMMIT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-saas", action="store_true", help="fail if optional SAAS is absent")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    for variable in ("MUSIC_DATA_ROOT", "CLAMP3_RUN_ROOT"):
        value = os.environ.get(variable)
        if value:
            print(f"ok: {variable}={Path(value).expanduser()}")
        else:
            failures.append(f"{variable} is not set")

    vendor = root / "vendor" / "clamp3"
    try:
        commit = subprocess.run(
            ["git", "-C", str(vendor), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        failures.append(f"cannot inspect CLaMP 3 submodule: {exc}")
    else:
        print(f"{'ok' if commit == UPSTREAM_COMMIT else 'FAIL'}: CLaMP 3 commit {commit}")
        if commit != UPSTREAM_COMMIT:
            failures.append(f"expected CLaMP 3 commit {UPSTREAM_COMMIT}")

    c2 = root / "models" / "clamp3-c2" / C2_CHECKPOINT
    wired = vendor / "code" / C2_CHECKPOINT
    if c2.is_file() and wired.is_file() and wired.resolve() == c2.resolve():
        print(f"ok: C2 checkpoint and symlink ({c2.stat().st_size} bytes)")
    else:
        failures.append("C2 checkpoint or protected symlink is unavailable")

    saas = root / "models" / "clamp3-saas" / SAAS_CHECKPOINT
    if saas.is_file():
        print(f"ok: optional SAAS checkpoint ({saas.stat().st_size} bytes)")
    elif args.require_saas:
        failures.append("optional SAAS checkpoint was required but is unavailable")
    else:
        print("optional: SAAS checkpoint is not installed (symbolic commands remain available)")

    required = ("numpy", "yaml", "sklearn", "torch", "transformers", "accelerate", "abctoolkit", "samplings")
    optional = ("pandas", "matplotlib", "pytest")
    for module in required:
        if importlib.util.find_spec(module):
            print(f"ok: Python module {module}")
        else:
            failures.append(f"Python module {module} is unavailable")
    for module in optional:
        state = "ok" if importlib.util.find_spec(module) else "missing (install from environment.yml)"
        print(f"optional: Python module {module}: {state}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("Environment is ready for C2 score and prompt embedding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
