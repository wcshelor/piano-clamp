#!/usr/bin/env python3
"""Write provenance and artifact metadata for a configured run."""

import sys

from piano_clamp.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["make-run-manifest", *sys.argv[1:]]))

