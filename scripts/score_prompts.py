#!/usr/bin/env python3
"""Compatibility wrapper for the score-prompts CLI command."""

import sys

from piano_clamp.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["score-prompts", *sys.argv[1:]]))

