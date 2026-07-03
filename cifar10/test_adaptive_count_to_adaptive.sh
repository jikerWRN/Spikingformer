#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 CHECKPOINT DATA_DIR [DEVICE]"
    echo "Example: $0 output/train/exp/model_best.pth.tar /data/cifar10 cuda:0"
    exit 1
fi

CHECKPOINT="$1"
DATA_DIR="$2"
DEVICE="${3:-cuda:0}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/test_adaptive_count_to_adaptive.py" \
    --checkpoint "${CHECKPOINT}" \
    --data-dir "${DATA_DIR}" \
    --device "${DEVICE}" \
    --config "${SCRIPT_DIR}/cifar10.yml" \
    --spike-percentile 0.7 0.8 0.9 0.99 \
    --spike-selected-percentile 0.99
