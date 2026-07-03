#!/bin/bash
# Task-2 Stage 1: MeanFlow one-step distillation — A100 / SAPI site
#
#   source paths.env && bash scripts/task2_train_meanflow.sh TEACHER_CKPT [BATCH] [ITERS] [OUTPUT_DIR]
#
# TEACHER_CKPT is the SFT checkpoint dir from Stage 0
# (e.g. $SIMVLA_CHECKPOINTS/task2_sft/ckpt-200000).
set -e
cd "$(dirname "$0")/.."

TEACHER_CKPT=${1:?usage: task2_train_meanflow.sh TEACHER_CKPT [BATCH] [ITERS] [OUTPUT_DIR]}
BATCH_SIZE=${2:-32}
ITERS=${3:-50000}
OUTPUT_DIR=${4:-${SIMVLA_CHECKPOINTS:-./runs}/task2_meanflow}

LIBERO_DATA_DIR=${LIBERO_DATASETS:-./datasets/metas}
TRAIN_METAS_PATH=${TRAIN_METAS_PATH:-${LIBERO_DATA_DIR}/libero_train.json}
NORM_STATS_PATH=${NORM_STATS_PATH:-./norm_stats/libero_norm.json}

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-0}
NUM_PROCESSES=${NUM_GPUS:-1}
export TF_CPP_MIN_LOG_LEVEL=2

echo "=== Task-2 MeanFlow distillation: teacher=$TEACHER_CKPT -> $OUTPUT_DIR ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes="$NUM_PROCESSES" \
    --main_process_port "${MAIN_PORT:-29514}" \
    --mixed_precision no \
    train_meanflow_distill.py \
    --teacher_ckpt "$TEACHER_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    --train_metas_path "$TRAIN_METAS_PATH" \
    --norm_stats_path "$NORM_STATS_PATH" \
    --action_mode libero_joint \
    --batch_size "$BATCH_SIZE" \
    --iters "$ITERS" \
    --learning_rate "${LR:-5e-5}" \
    --meanflow_ratio "${MEANFLOW_RATIO:-0.5}" \
    --warmup_steps 1000 \
    --save_interval 5000 \
    --num_workers "${NUM_WORKERS:-4}" \
    ${EXTRA_ARGS:-}
