#!/usr/bin/env bash
# Queue this CPU-side Bach readiness audit with:
#   sbatch jobs/audit_bach_readiness.slurm

#SBATCH --job-name=bach-readiness
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail

set +u
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-piano-clamp-env}"
set -u

cd "${ROOT_DIR:-/share/users/student/w/wshelor/piano-clamp}"

mkdir -p logs

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
export PIANO_CLAMP_CORPUS_ROOT="${PIANO_CLAMP_CORPUS_ROOT:-/share/users/student/w/wshelor/datasets/classical-performance-corpus}"

CONFIG="${CONFIG:-configs/embedding_config_bach_symbolic.yaml}"
ADAPTER_ROOT="${ADAPTER_ROOT:-data/cpc_symbolic_adapter/bach-v0}"
REPORT_DIR="${REPORT_DIR:-../piano-clamp-runs/bach_symbolic_v0/analysis/reports}"

echo "=== Bach readiness audit ==="
echo "date: $(date -Is)"
echo "host: $(hostname)"
echo "root: $(pwd)"
echo "config: ${CONFIG}"
echo "adapter_root: ${ADAPTER_ROOT}"
echo "corpus_root: ${PIANO_CLAMP_CORPUS_ROOT}"

if [[ ! -f "${ADAPTER_ROOT}/piano_clamp_manifest.csv" ]]; then
  echo "Bach adapter manifest not found; building it now."
  python scripts/prepare_cpc_symbolic_adapter.py \
    --corpus-root "${PIANO_CLAMP_CORPUS_ROOT}" \
    --output-root "${ADAPTER_ROOT}" \
    --composer Bach
fi

mkdir -p "${REPORT_DIR}"

python scripts/check_installation.py \
  --config "${CONFIG}" \
  --json

python scripts/audit_study_readiness.py \
  --config "${CONFIG}" \
  --output "${REPORT_DIR}/bach_study_readiness.json"

echo "Readiness report written to ${REPORT_DIR}/bach_study_readiness.json"
