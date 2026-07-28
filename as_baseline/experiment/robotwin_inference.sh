#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DISTCP_MODEL_DIR="/data1/fangxuebin/models/cosmos_predict2_action_conditioned/robotwin/robotwin_14d_finetune_pretrained_4gpu_fc1_random_fc2_zero/checkpoints/iter_000002000/model"
CONVERTED_DIR="as_baseline/checkpoints/robotwin_14d_finetune_pretrained_4gpu_fc1_random_fc2_zero_iter2000"
CONVERTED_PT="${CONVERTED_DIR}/model.pt"

if [[ ! -f "${CONVERTED_PT}" ]]; then
  mkdir -p "${CONVERTED_DIR}"
  .venv/bin/python scripts/convert_distcp_to_pt.py "${DISTCP_MODEL_DIR}" "${CONVERTED_DIR}" 
fi

.venv/bin/python examples/action_conditioned.py \
  -i as_baseline/inference_params.json \
  -o outputs/robotwin_14d_finetune_pretrained_4gpu_fc1_random_fc2_zero_iter2000 \
  --context_parallel_size 1 \
  --experiment robotwin_14d_finetune \
  --checkpoint_path "${CONVERTED_PT}"

# CUDA_VISIBLE_DEVICES=4,5,6,7  bash as_baseline/experiment/robotwin_inference.sh
