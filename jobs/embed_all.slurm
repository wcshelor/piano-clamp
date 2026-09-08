#!/bin/bash
# Edit these commented examples for the target cluster:
##SBATCH --partition=REPLACE_WITH_PARTITION
##SBATCH --account=REPLACE_WITH_ACCOUNT
##SBATCH --gres=gpu:1
#SBATCH --job-name=piano-clamp-all
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail

# Activate the site-specific environment before submission or add its command here.
: "${PIANO_CLAMP_CORPUS_ROOT:?Set PIANO_CLAMP_CORPUS_ROOT to the read-only corpus checkout}"

python scripts/run_all_embeddings.py \
  --config configs/embedding_config.yaml \
  --device cuda \
  --text \
  --symbolic \
  --audio \
  --passages \
  --index \
  --similarities \
  --validate

