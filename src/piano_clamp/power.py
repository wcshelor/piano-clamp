"""Optional power-management helpers for long-running local jobs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from typing import Iterator


_VALID_CAFFEINATE_FLAGS = set("dimsu")


def add_caffeinate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--caffeinate",
        action="store_true",
        help="on macOS, run a background caffeinate helper tied to this process",
    )
    parser.add_argument(
        "--caffeinate-flags",
        default="is",
        help="macOS caffeinate flags to use with --caffeinate (default: is)",
    )


def validate_caffeinate_flags(value: str) -> str:
    flags = "".join(dict.fromkeys(str(value).strip()))
    if not flags:
        raise ValueError("caffeinate flags must not be empty")
    invalid = sorted(set(flags) - _VALID_CAFFEINATE_FLAGS)
    if invalid:
        raise ValueError(
            "unsupported caffeinate flag(s): " + ", ".join(invalid) + "; allowed: d, i, m, s, u"
        )
    return flags


@contextmanager
def maybe_caffeinate(enabled: bool, *, flags: str = "is") -> Iterator[None]:
    """Keep a macOS machine awake while the current process is alive."""

    if not enabled:
        yield
        return
    flags = validate_caffeinate_flags(flags)
    if sys.platform != "darwin":
        print("note: --caffeinate ignored because this is not macOS", file=sys.stderr, flush=True)
        yield
        return
    executable = shutil.which("caffeinate")
    if not executable:
        raise RuntimeError("macOS caffeinate command was requested but is not available on PATH")
    process = subprocess.Popen(
        [executable, f"-{flags}", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        yield
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
