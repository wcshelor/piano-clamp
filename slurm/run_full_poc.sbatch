#!/usr/bin/env bash
#SBATCH --job-name=pclamp-full
#SBATCH --partition=REPLACE_WITH_PARTITION
#SBATCH --account=REPLACE_WITH_ACCOUNT
#SBATCH --gres=gpu:REPLACE_WITH_GPU_COUNT
#SBATCH --mem=REPLACE_WITH_MEMORY
#SBATCH --time=REPLACE_WITH_TIME
#SBATCH --output=REPLACE_WITH_LOG_DIRECTORY/%x-%j.out

set -euo pipefail

# Site-specific module setup, if required:
# module load REPLACE_WITH_CUDA_MODULE
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate piano-clamp-env

: "${MUSIC_DATA_ROOT:?Set MUSIC_DATA_ROOT to the external data repository}"
: "${CLAMP3_RUN_ROOT:?Set CLAMP3_RUN_ROOT to the external run root}"

CONFIG=configs/experiment_chopin_mozart.yaml
python -m piano_clamp.cli validate-data --config "${CONFIG}"
python -m piano_clamp.cli extract-features --config "${CONFIG}"
python -m piano_clamp.cli embed-scores --config "${CONFIG}"
python -m piano_clamp.cli embed-prompts --config "${CONFIG}"
python -m piano_clamp.cli score-prompts --config "${CONFIG}"
python -m piano_clamp.cli analyze --config "${CONFIG}"
python -m piano_clamp.cli make-run-manifest --config "${CONFIG}"
