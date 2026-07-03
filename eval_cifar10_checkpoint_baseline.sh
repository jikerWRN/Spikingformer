#!/usr/bin/env bash
set -euo pipefail

MODEL_FILE="${1:-model_4layers_baseline_ILIF01_count_test}"
DATA_DIR="${2:-/home/wangyufei/dataset}"
CHECKPOINT="${3:-/home/wangyufei/code/SNNTransformer/Spikingformer/output/train/model_4layers_baseline_ILIF01_count-20260701-081419/model_best.pth.tar}"
DEVICE="${4:-cuda:0}"

if [ "$#" -gt 0 ]; then shift 1; fi
if [ "$#" -gt 0 ]; then shift 1; fi
if [ "$#" -gt 0 ]; then shift 1; fi
if [ "$#" -gt 0 ]; then shift 1; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/cifar10"

python eval_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --data-dir "$DATA_DIR" \
  --model-file "$MODEL_FILE" \
  --device "$DEVICE" \
  "$@"
