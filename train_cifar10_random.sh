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


# ------------------------------# -------------------------------- # -------------------
随机模型


# ----------------------------------------------------------------
20260617

CUDA_VISIBLE_DEVICES=1 python cifar10/train.py -c cifar10/cifar10.yml --model-file model_4layers_baseline -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100
b

CUDA_VISIBLE_DEVICES=0 python cifar10/train.py -c cifar10/cifar10.yml --model-file model_4layers_random -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100
a


# ----------------------------------------------------------------
20260618

CUDA_VISIBLE_DEVICES=0 python cifar10/train.py -c cifar10/cifar10.yml --model-file model_4layers_random_RF -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100





# ------------------------------# -------------------------------- # -------------------
Fuzzy 神经元

# ----------------------------------------------------------------
20260618


##########################################################################

CUDA_VISIBLE_DEVICES=0 python cifar10/train.py -c cifar10/cifar10.yml --model-file model_4layers_baseline_ILIF01 -data-dir /home/wangyufei/dataset --output /home/wangyufei/code/SNNTransformer/Spikingformer/output/train --experiment Spikingformer-4-384-random --iter-eval-start-epoch 350 --iter-eval-interval 100
