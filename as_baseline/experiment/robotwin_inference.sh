#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

DISTCP_MODEL_DIR="/tmp/imaginaire4-output/cosmos_predict2_action_conditioned/robotwin/robotwin_14d_finetune_pretrained_2gpu_run1/checkpoints/iter_000009000/model"
CONVERTED_DIR="as_baseline/checkpoints/robotwin_iter9000"
CONVERTED_PT="${CONVERTED_DIR}/model.pt"

if [[ ! -f "${CONVERTED_PT}" ]]; then
  mkdir -p "${CONVERTED_DIR}"
  .venv/bin/python scripts/convert_distcp_to_pt.py "${DISTCP_MODEL_DIR}" "${CONVERTED_DIR}" --no-ema
fi

CUDA_LAUNCH_BLOCKING=1 .venv/bin/python examples/action_conditioned.py \
  -i as_baseline/inference_params.json \
  -o outputs/robotwin_iter9000 \
  --context_parallel_size 1 \
  --experiment robotwin_14d_finetune \
  --checkpoint_path "${CONVERTED_PT}"

# CUDA_LAUNCH_BLOCKING=1 bash as_baseline/experiment/robotwin_inference.sh
