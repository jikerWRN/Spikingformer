#!/usr/bin/env bash
set -euo pipefail

MODEL_FILE="${1:-model_4layers_random}"
DATA_DIR="${2:-/home/wangyufei/dataset}"
CHECKPOINT="${3:-./output/train/model_4layers_random/model_best.pth.tar}"
DEVICE="${4:-cuda:0}"
BATCH_SIZE="${5:-64}"
WORKERS="${6:-8}"
TTA="${7:-0}"
NO_PREFETCHER="${8:-0}"
NO_AMP="${9:-0}"
CHANNELS_LAST="${10:-0}"

if [ "$#" -gt 0 ]; then shift 1; fi
if [ "$#" -gt 0 ]; then shift 1; fi
if [ "$#" -gt 0 ]; then shift 1; fi
if [ "$#" -gt 0 ]; then shift 1; fi
if [ "$#" -gt 0 ]; then shift 1; fi
if [ "$#" -gt 0 ]; then shift 1; fi
if [ "$#" -gt 0 ]; then shift 1; fi
if [ "$#" -gt 0 ]; then shift 1; fi
if [ "$#" -gt 0 ]; then shift 1; fi
if [ "$#" -gt 0 ]; then shift 1; fi

EXTRA_ARGS=()
if [ "$NO_PREFETCHER" = "1" ]; then
  EXTRA_ARGS+=("--no-prefetcher")
fi
if [ "$NO_AMP" = "1" ]; then
  EXTRA_ARGS+=("--no-amp")
fi
if [ "$CHANNELS_LAST" = "1" ]; then
  EXTRA_ARGS+=("--channels-last")
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/cifar10"

python eval_checkpoint.py \
  --checkpoint "$CHECKPOINT" \
  --data-dir "$DATA_DIR" \
  --model-file "$MODEL_FILE" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --workers "$WORKERS" \
  --tta "$TTA" \
  "${EXTRA_ARGS[@]}" \
  "$@"
