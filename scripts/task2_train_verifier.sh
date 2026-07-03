#!/bin/bash
# Task-2 Stage 2: Energy verifier training — A800 / YYK site
#
#   source paths.env && bash scripts/task2_train_verifier.sh POLICY_CKPT GENERATOR_CKPT [BATCH] [ITERS] [OUTPUT_DIR]
#
# POLICY_CKPT    = SFT checkpoint (frozen features), shipped from the A100 site
# GENERATOR_CKPT = MeanFlow one-step checkpoint (negatives), shipped from the A100 site
#
# Asynchronous co-training: whenever a newer generator checkpoint arrives from
# the A100 site, just re-run this script pointing at it (optionally resuming
# the verifier weights via EXTRA_ARGS).
set -e
cd "$(dirname "$0")/.."

POLICY_CKPT=${1:?usage: task2_train_verifier.sh POLICY_CKPT GENERATOR_CKPT [BATCH] [ITERS] [OUTPUT_DIR]}
GENERATOR_CKPT=${2:?usage: task2_train_verifier.sh POLICY_CKPT GENERATOR_CKPT [BATCH] [ITERS] [OUTPUT_DIR]}
BATCH_SIZE=${3:-32}
ITERS=${4:-50000}
OUTPUT_DIR=${5:-${SIMVLA_CHECKPOINTS:-./runs}/task2_verifier}

LIBERO_DATA_DIR=${LIBERO_DATASETS:-./datasets/metas}
TRAIN_METAS_PATH=${TRAIN_METAS_PATH:-${LIBERO_DATA_DIR}/libero_train.json}
NORM_STATS_PATH=${NORM_STATS_PATH:-./norm_stats/libero_norm.json}

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-0}
export TF_CPP_MIN_LOG_LEVEL=2

echo "=== Task-2 verifier: policy=$POLICY_CKPT generator=$GENERATOR_CKPT -> $OUTPUT_DIR ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes="${NUM_GPUS:-1}" \
    --main_process_port "${MAIN_PORT:-29524}" \
    --mixed_precision no \
    train_verifier.py \
    --policy_ckpt "$POLICY_CKPT" \
    --generator_ckpt "$GENERATOR_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --train_metas_path "$TRAIN_METAS_PATH" \
    --norm_stats_path "$NORM_STATS_PATH" \
    --batch_size "$BATCH_SIZE" \
    --iters "$ITERS" \
    --num_neg_gen "${NUM_NEG_GEN:-8}" \
    --num_neg_perturb "${NUM_NEG_PERTURB:-4}" \
    --num_neg_shuffle "${NUM_NEG_SHUFFLE:-4}" \
    --temperature "${TEMPERATURE:-0.1}" \
    --num_workers "${NUM_WORKERS:-4}" \
    ${EXTRA_ARGS:-}
