#!/usr/bin/env bash
# CPU-capable validation job for piano-clamp.
# Add cluster-specific resources at submit time, for example:
#   sbatch --partition=gpu --account=YOUR_ACCOUNT --gres=gpu:1 jobs/validate_hpc_verbose.slurm
# or:
#   sbatch --partition=cpu --account=YOUR_ACCOUNT jobs/validate_hpc_verbose.slurm
#
# Required environment:
#   PIANO_CLAMP_CORPUS_ROOT=/path/to/classical-performance-corpus
#
# Optional environment:
#   CONDA_ENV_NAME=piano-clamp-env
#   RUN_PYTEST=1
#   RUN_PASSAGE_SMOKE=1

#SBATCH --job-name=piano-clamp-validate
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out

set -Eeuo pipefail
set -x
export PS4='+ [${BASH_SOURCE##*/}:${LINENO}] '

on_error() {
    local exit_code=$?
    local line_no=${1:-unknown}
    echo
    echo "============================================================"
    echo "piano-clamp HPC validation failed"
    echo "date: $(date -Is)"
    echo "host: $(hostname)"
    echo "pwd: $(pwd)"
    echo "line: ${line_no}"
    echo "exit_code: ${exit_code}"
    echo "job_id: ${SLURM_JOB_ID:-unset}"
    echo "============================================================"
    echo
    echo "Recent artifacts:"
    find "${JOB_DIR:-logs/job-manual}" -maxdepth 2 -type f 2>/dev/null | sort || true
    echo
    echo "Last 200 lines from bash trace:"
    tail -n 200 "${TRACE_LOG:-/dev/null}" 2>/dev/null || true
    echo
    exit "${exit_code}"
}

trap 'on_error $LINENO' ERR

ROOT_DIR="${ROOT_DIR:-$PWD}"
cd "${ROOT_DIR}"

mkdir -p logs
JOB_DIR="logs/job-${SLURM_JOB_ID:-manual}"
mkdir -p "${JOB_DIR}"
TRACE_LOG="${JOB_DIR}/bash-trace.log"
exec > >(tee -a "${JOB_DIR}/job.log")
exec 2> >(tee -a "${JOB_DIR}/job.err" >&2)
exec 19>>"${TRACE_LOG}"
export BASH_XTRACEFD=19

echo "=== piano-clamp HPC validation ==="
echo "date: $(date -Is)"
echo "host: $(hostname)"
echo "root_dir: ${ROOT_DIR}"
echo "job_dir: ${JOB_DIR}"
echo "slurm_job_id: ${SLURM_JOB_ID:-unset}"
echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-unset}"

if [[ -n "${MODULESHOME:-}" ]] && command -v module >/dev/null 2>&1; then
    module list 2>&1 || true
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-piano-clamp-env}"
if command -v micromamba >/dev/null 2>&1; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate "${CONDA_ENV_NAME}"
elif command -v mamba >/dev/null 2>&1; then
    eval "$(mamba shell hook --shell bash)"
    mamba activate "${CONDA_ENV_NAME}"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV_NAME}"
else
    echo "Could not find micromamba, mamba, or conda on PATH."
    exit 1
fi

: "${PIANO_CLAMP_CORPUS_ROOT:?Set PIANO_CLAMP_CORPUS_ROOT to the read-only classical-performance-corpus checkout}"
export MUSIC_DATA_ROOT="${MUSIC_DATA_ROOT:-$PIANO_CLAMP_CORPUS_ROOT}"
export CLAMP3_RUN_ROOT="${CLAMP3_RUN_ROOT:-$ROOT_DIR}"
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM=false

python --version
which python
python -m pip --version

python - <<'PY'
import importlib
modules = [
    "torch",
    "transformers",
    "accelerate",
    "music21",
    "soundfile",
    "numpy",
    "yaml",
    "sklearn",
    "pytest",
]
for name in modules:
    try:
        module = importlib.import_module(name)
        print(f"{name}: ok version={getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}: FAIL {exc}")
PY

nvidia-smi || true

HAS_NVIDIA_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    HAS_NVIDIA_GPU=1
fi
echo "has_nvidia_gpu: ${HAS_NVIDIA_GPU}"

python scripts/verify_environment.py 2>&1 | tee "${JOB_DIR}/verify_environment.txt"
python scripts/check_installation.py --config configs/embedding_config.yaml --json 2>&1 | tee "${JOB_DIR}/check_installation.json"

if [[ "${RUN_PYTEST:-1}" == "1" ]]; then
    python -m pytest -vv -ra 2>&1 | tee "${JOB_DIR}/pytest.txt"
fi

python scripts/smoke_test.py \
    --config configs/embedding_config.yaml \
    --device cpu \
    --output "${JOB_DIR}/smoke-cpu" \
    2>&1 | tee "${JOB_DIR}/smoke-cpu.txt"

if [[ "${RUN_PASSAGE_SMOKE:-1}" == "1" ]]; then
    python scripts/smoke_test_passages.py \
        --config configs/embedding_config.yaml \
        --device cpu \
        --modality symbolic \
        --mode 16 \
        --limit 3 \
        --output "${JOB_DIR}/passages-cpu" \
        2>&1 | tee "${JOB_DIR}/passages-cpu.txt"
fi

if [[ "${HAS_NVIDIA_GPU}" == "1" ]]; then
    python scripts/smoke_test.py \
        --config configs/embedding_config.yaml \
        --device cuda \
        --output "${JOB_DIR}/smoke-cuda" \
        2>&1 | tee "${JOB_DIR}/smoke-cuda.txt"

    if [[ "${RUN_PASSAGE_SMOKE:-1}" == "1" ]]; then
        python scripts/smoke_test_passages.py \
            --config configs/embedding_config.yaml \
            --device cuda \
            --modality symbolic \
            --mode 16 \
            --limit 3 \
            --output "${JOB_DIR}/passages-cuda" \
            2>&1 | tee "${JOB_DIR}/passages-cuda.txt"
    fi
else
    echo "Skipping CUDA smoke tests because no NVIDIA GPU is visible on this node."
fi

echo
echo "Validation job completed successfully."
find "${JOB_DIR}" -maxdepth 2 -type f | sort
