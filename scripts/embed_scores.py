#!/usr/bin/env python3
"""Compatibility wrapper for the embed-scores CLI command."""

import sys

from piano_clamp.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["embed-scores", *sys.argv[1:]]))

