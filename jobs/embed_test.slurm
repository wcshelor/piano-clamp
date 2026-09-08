#!/bin/bash
# This smoke job is CPU-capable. Uncomment and edit cluster-specific directives.
##SBATCH --partition=REPLACE_WITH_PARTITION
##SBATCH --account=REPLACE_WITH_ACCOUNT
#SBATCH --job-name=piano-clamp-smoke
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail

: "${PIANO_CLAMP_CORPUS_ROOT:?Set PIANO_CLAMP_CORPUS_ROOT to the read-only corpus checkout}"
python scripts/check_installation.py --config configs/embedding_config.yaml
python scripts/smoke_test.py --config configs/embedding_config.yaml --device cpu

