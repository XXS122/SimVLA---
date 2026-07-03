#!/bin/bash
# Task-2 Stage 0: SFT baseline (teacher) — A100 / SAPI site
#
# Single-node training driven by paths.env:
#   source paths.env && bash scripts/task2_train_sft.sh [BATCH_SIZE] [OUTPUT_DIR]
#
# Env vars used (from paths.env):
#   SIMVLA_SMOLVLM_MODEL  local SmolVLM path or HF repo
#   LIBERO_DATASETS       LIBERO data root
#   SIMVLA_CHECKPOINTS    checkpoint output root
#   SIMVLA_RESUME_CKPT    optional resume checkpoint
#   CUDA_DEVICES / NUM_GPUS
set -e
cd "$(dirname "$0")/.."

BATCH_SIZE=${1:-64}
OUTPUT_DIR=${2:-${SIMVLA_CHECKPOINTS:-./runs}/task2_sft}
RESUME_CKPT=${SIMVLA_RESUME_CKPT:-""}

SMOLVLM_MODEL=${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}
LIBERO_DATA_DIR=${LIBERO_DATASETS:-./datasets/metas}
TRAIN_METAS_PATH=${TRAIN_METAS_PATH:-${LIBERO_DATA_DIR}/libero_train.json}
NORM_STATS_PATH=${NORM_STATS_PATH:-./norm_stats/libero_norm.json}

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-0}
NUM_PROCESSES=${NUM_GPUS:-1}
export TF_CPP_MIN_LOG_LEVEL=2

# Derived files
if [ ! -f "$TRAIN_METAS_PATH" ]; then
    echo "Creating training metadata at $TRAIN_METAS_PATH ..."
    python create_libero_meta.py \
        --data_dir "$LIBERO_DATA_DIR" \
        --subsets libero_10 libero_goal libero_object libero_spatial libero_90 \
        --output "$TRAIN_METAS_PATH"
fi
if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "Computing normalization statistics at $NORM_STATS_PATH ..."
    python compute_libero_norm_stats.py \
        --data_dir "$LIBERO_DATA_DIR" \
        --subsets libero_10 libero_goal libero_object libero_spatial libero_90 \
        --output "$NORM_STATS_PATH"
fi

ARGS="--output_dir ${OUTPUT_DIR} \
    --train_metas_path ${TRAIN_METAS_PATH} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode libero_joint \
    --batch_size ${BATCH_SIZE} \
    --learning_rate 1e-4 \
    --learning_coef 0.1 \
    --num_actions 10 \
    --iters ${ITERS:-200000} \
    --warmup_steps 0 \
    --freeze_steps 1000 \
    --hidden_size 768 \
    --depth 12 \
    --num_heads 12 \
    --num_workers ${NUM_WORKERS:-4} \
    --save_interval 10000 \
    --log_interval 20 \
    --image_size 384 \
    --norm_stats_path ${NORM_STATS_PATH} \
    --max_grad_norm 1.0"

if [ -n "$RESUME_CKPT" ]; then
    ARGS="$ARGS --models $RESUME_CKPT --resume"
    echo "Resuming from $RESUME_CKPT"
fi

echo "=== Task-2 SFT teacher: gpus=$CUDA_VISIBLE_DEVICES x$NUM_PROCESSES batch=$BATCH_SIZE -> $OUTPUT_DIR ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes="$NUM_PROCESSES" \
    --main_process_port "${MAIN_PORT:-29504}" \
    --mixed_precision bf16 \
    train_smolvlm.py $ARGS
