#!/bin/bash
# Queue this job with matching inputs, for example:
#   sbatch --export=ALL,CORPUS_ROOT=/path/to/corpus,PASSAGE_BUNDLE=/path/to/passages,PROMPT_BUNDLE=/path/to/text,OUTPUT_DIR=/path/to/output jobs/feature_embedding_correlation_analysis.slurm

#SBATCH --partition=gpu
#SBATCH --job-name=feature-embed-corr
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%x-%j.out

set -euo pipefail

set +u
source /share/users/student/w/wshelor/miniforge3/etc/profile.d/conda.sh
conda activate piano-clamp-env
set -u

cd /share/users/student/w/wshelor/piano-clamp

: "${PASSAGE_BUNDLE:?Set PASSAGE_BUNDLE to the matching passage embedding bundle directory}"
: "${PROMPT_BUNDLE:?Set PROMPT_BUNDLE to the matching prompt embedding bundle directory}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for analysis outputs}"
: "${FEATURE_TABLE:=}"
: "${CORPUS_ROOT:=}"
: "${ANALYSIS_UNIT:=work_id}"
: "${PCA_COMPONENTS:=32}"
: "${EMBEDDING_DIMENSIONS:=768}"
: "${TOP_DIMENSIONS_PER_FEATURE:=20}"

FEATURE_ARGS=()
if [[ -n "${FEATURE_TABLE}" ]]; then
  FEATURE_ARGS=(--feature-table "${FEATURE_TABLE}")
else
  : "${CORPUS_ROOT:?Set CORPUS_ROOT unless FEATURE_TABLE is provided}"
  FEATURE_ARGS=(--corpus-root "${CORPUS_ROOT}")
fi

python scripts/feature_embedding_correlation_analysis.py \
  "${FEATURE_ARGS[@]}" \
  --passage-bundle "${PASSAGE_BUNDLE}" \
  --prompt-bundle "${PROMPT_BUNDLE}" \
  --output-dir "${OUTPUT_DIR}" \
  --unit "${ANALYSIS_UNIT}" \
  --pca-components "${PCA_COMPONENTS}" \
  --embedding-dimensions "${EMBEDDING_DIMENSIONS}" \
  --top-dimensions-per-feature "${TOP_DIMENSIONS_PER_FEATURE}"
