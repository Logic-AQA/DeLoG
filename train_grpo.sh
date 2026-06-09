#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$SCRIPT_DIR}
PYTHON=${PYTHON:-python}

GPU=${GPU:-0}
MODEL_PATH=${MODEL_PATH:-saves/sft-best-asqa-repro}
DATA_PATH=${DATA_PATH:-data_grpo.json}
OUTPUT_DIR=${OUTPUT_DIR:-saves/grpo-doc-refusal-reward-fix-20260522}
WANDB_NAME=${WANDB_NAME:-grpo-doc-refusal-reward-fix-20260522}
MAX_STEPS=${MAX_STEPS:-1800}
NUM_GENERATIONS=${NUM_GENERATIONS:-16}
SAVE_STEPS=${SAVE_STEPS:-300}

cd "$ROOT"

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "DATA_PATH does not exist: $DATA_PATH" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" \
  train_grpo.py \
  --model-path "$MODEL_PATH" \
  --data-path "$DATA_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --wandb-name "$WANDB_NAME" \
  --max-steps "$MAX_STEPS" \
  --num-generations "$NUM_GENERATIONS" \
  --save-steps "$SAVE_STEPS" \
  --debug-reward \
  --debug-reward-batches "$MAX_STEPS" \
  --debug-reward-samples 0
