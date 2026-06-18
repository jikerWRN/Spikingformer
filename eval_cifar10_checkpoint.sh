#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <checkpoint.pth.tar> <cifar10_root> [model_file] [device] [extra args...]"
  echo "Example: $0 ./checkpoints/model_best.pth.tar ./data model_4layers_random cuda:0 --download"
  exit 1
fi

CHECKPOINT="$1"
DATA_DIR="$2"
MODEL_FILE="${3:-model}"
DEVICE="${4:-cuda:0}"

shift 2
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
