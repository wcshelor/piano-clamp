#!/bin/bash
# Queue this job with:
#   sbatch jobs/embed_passages_chopin.slurm

#SBATCH --partition=gpu
#SBATCH --job-name=chopin-passages
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate piano-clamp-env
set -u

cd /share/users/student/w/wshelor/piano-clamp

python scripts/embed_passages.py \
  --config configs/embedding_config.yaml \
  --device cuda \
  --modality symbolic \
  --composer Chopin \
  --mode 4 \
  --mode 8 \
  --mode 16 \
  --overlap-half \
  --force \
  --status-every 10 \
  --chunk-size 25
