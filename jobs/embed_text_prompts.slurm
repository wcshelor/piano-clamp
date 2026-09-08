#!/bin/bash
# Queue this job with:
#   sbatch jobs/embed_text_prompts.slurm

#SBATCH --partition=gpu
#SBATCH --job-name=text-prompts
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate piano-clamp-env
set -u

cd /share/users/student/w/wshelor/piano-clamp

python scripts/embed_text_prompts.py \
  --config configs/embedding_config.yaml \
  --device cuda \
  --force
