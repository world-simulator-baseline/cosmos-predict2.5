#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DISTCP_MODEL_DIR="/data1/fangxuebin/models/cosmos_predict2_action_conditioned/robotwin/robotwin_14d_finetune_pretrained_4gpu_5task_zero/checkpoints/iter_000004000/model"
CONVERTED_DIR="as_baseline/checkpoints/robotwin/robotwin_14d_finetune_pretrained_4gpu_5task_zero_iter4000"
CONVERTED_PT="${CONVERTED_DIR}/model.pt"
CONVERSION_MARKER="${CONVERTED_DIR}/.conversion_complete"

if [[ ! -f "${DISTCP_MODEL_DIR}/.metadata" ]]; then
  echo "Distributed model checkpoint not found: ${DISTCP_MODEL_DIR}/.metadata" >&2
  exit 1
fi

if [[ ! -f "${CONVERTED_PT}" || ! -f "${CONVERSION_MARKER}" ]]; then
  mkdir -p "${CONVERTED_DIR}"
  .venv/bin/python scripts/convert_distcp_to_pt.py "${DISTCP_MODEL_DIR}" "${CONVERTED_DIR}"
  touch "${CONVERSION_MARKER}"
fi

.venv/bin/python examples/action_conditioned.py \
  -i as_baseline/inference_params.json \
  -o outputs/robotwin_14d_finetune_pretrained_4gpu_5task_zero_iter4000 \
  --context_parallel_size 1 \
  --experiment robotwin_14d_finetune \
  --checkpoint_path "${CONVERTED_PT}"

# CUDA_VISIBLE_DEVICES=4  bash as_baseline/experiment/robotwin_inference.sh
