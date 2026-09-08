#!/usr/bin/env python3
"""Compatibility wrapper for the validate-data CLI command."""

import sys

from piano_clamp.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["validate-data", *sys.argv[1:]]))

