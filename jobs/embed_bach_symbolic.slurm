#!/bin/bash
# Queue this job with:
#   sbatch jobs/embed_bach_symbolic.slurm

#SBATCH --partition=gpu
#SBATCH --job-name=bach-symbolic
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

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export PIANO_CLAMP_CORPUS_ROOT="${PIANO_CLAMP_CORPUS_ROOT:-/share/users/student/w/wshelor/datasets/classical-performance-corpus}"

CONFIG=configs/embedding_config_bach_symbolic.yaml
ADAPTER_ROOT=data/cpc_symbolic_adapter/bach-v0

mkdir -p logs

if [[ ! -f "${ADAPTER_ROOT}/piano_clamp_manifest.csv" ]]; then
  python scripts/prepare_cpc_symbolic_adapter.py \
    --corpus-root "${PIANO_CLAMP_CORPUS_ROOT}" \
    --output-root "${ADAPTER_ROOT}" \
    --composer Bach
fi

python scripts/run_all_embeddings.py \
  --config "${CONFIG}" \
  --device cuda \
  --text \
  --symbolic \
  --passages \
  --passage-source-material score \
  --force
