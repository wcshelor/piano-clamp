#!/usr/bin/env python3
"""Compatibility wrapper for the extract-features CLI command."""

import sys

from piano_clamp.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["extract-features", *sys.argv[1:]]))

