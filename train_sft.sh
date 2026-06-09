#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$SCRIPT_DIR}
DEEPSPEED=${DEEPSPEED:-deepspeed}

GPUS=${GPUS:-0,1}
NUM_GPUS=${NUM_GPUS:-2}
MODEL_PATH=${MODEL_PATH:-}
DATA_PATH=${DATA_PATH:-data_sft.json}
SAVE_PATH=${SAVE_PATH:-saves/sft-best-asqa-repro}
OUTPUT_MODE=${OUTPUT_MODE:-think_cite_plan_answer}

cd "$ROOT"

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH must point to a local base model checkpoint" >&2
  exit 1
fi

if [[ ! -e "$MODEL_PATH" ]]; then
  echo "MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$DATA_PATH" ]]; then
  echo "DATA_PATH does not exist: $DATA_PATH" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$GPUS" "$DEEPSPEED" --num_gpus "$NUM_GPUS" \
  train_sft.py \
  --model_path "$MODEL_PATH" \
  --data_path "$DATA_PATH" \
  --save_path "$SAVE_PATH" \
  --output_mode "$OUTPUT_MODE"
