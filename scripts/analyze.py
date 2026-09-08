#!/usr/bin/env python3
"""Compatibility wrapper for the preregistered analyze CLI command."""

import sys

from piano_clamp.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["analyze", *sys.argv[1:]]))

