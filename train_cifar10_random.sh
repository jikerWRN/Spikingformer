#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_FILE="${1:-model_4layers_random}"
DATA_DIR="${2:-/home/wangyufei/dataset}"
OUTPUT_DIR="${3:-./output/train}"
EXPERIMENT="${4:-Spikingformer-4-384-random}"
ITER_EVAL_START_EPOCH="${5:--1}"
ITER_EVAL_INTERVAL="${6:-0}"

cd "$SCRIPT_DIR/cifar10"

python train.py \
  -c cifar10.yml \
  --model-file "$MODEL_FILE" \
  -data-dir "$DATA_DIR" \
  --output "$OUTPUT_DIR" \
  --experiment "$EXPERIMENT" \
  --iter-eval-start-epoch "$ITER_EVAL_START_EPOCH" \
  --iter-eval-interval "$ITER_EVAL_INTERVAL"

