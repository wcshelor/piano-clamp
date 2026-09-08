#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "saas" ]]; then
  echo "Usage: bash models/download_checkpoint.sh saas" >&2
  echo "Only the optional SAAS checkpoint is downloaded; C2 is already installed." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILENAME="weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
DESTINATION="${SCRIPT_DIR}/clamp3-saas/${FILENAME}"
URL="https://huggingface.co/sander-wood/clamp3/resolve/main/${FILENAME}"

mkdir -p "${SCRIPT_DIR}/clamp3-saas"
if [[ -e "${DESTINATION}" ]]; then
  echo "Refusing to overwrite existing checkpoint: ${DESTINATION}" >&2
  exit 1
fi
if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for this explicit download." >&2
  exit 1
fi

echo "Downloading optional CLaMP 3 SAAS checkpoint to ${DESTINATION}"
curl --fail --location --continue-at - --output "${DESTINATION}.part" "${URL}"
mv "${DESTINATION}.part" "${DESTINATION}"
echo "Download complete. The checkpoint remains ignored by Git."

