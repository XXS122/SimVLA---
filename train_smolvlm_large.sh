#!/bin/bash
# SimVLA Training Script for LIBERO (Large Model)
# 
# Key features:
#   - 384x384 image resolution (SmolVLM requirement)
#   - All views processed together by VLM (no aux_visual_inputs)
#   - Larger action transformer configuration

set -e

# =============================================================================
# Command line arguments (with defaults)
# =============================================================================

BATCH_SIZE=${1:-64}
LEARNING_COEF=${2:-0.1}
# Default output dir falls back to $SIMVLA_CHECKPOINTS (from paths.env) if set
OUTPUT_DIR=${3:-${SIMVLA_CHECKPOINTS:-./runs/simvla_libero_large}}
# Default resume checkpoint falls back to $SIMVLA_RESUME_CKPT (from paths.env)
RESUME_CKPT=${4:-${SIMVLA_RESUME_CKPT:-""}}

# GPU configuration (read from paths.env: CUDA_DEVICES / NUM_GPUS)
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES:-4,5,6,7}
NUM_PROCESSES=${NUM_GPUS:-4}

echo "Training parameters:"
echo "   batch_size: $BATCH_SIZE"
echo "   learning_coef: $LEARNING_COEF"
echo "   output_dir: $OUTPUT_DIR"
echo "   resume_ckpt: ${RESUME_CKPT:-'None (training from scratch)'}"
echo "   CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "   num_processes: $NUM_PROCESSES"

# Suppress TensorFlow logs
export TF_CPP_MIN_LOG_LEVEL=2

# =============================================================================
# Path configuration (read from paths.env where available)
# =============================================================================
LIBERO_DATA_DIR="${LIBERO_DATASETS:-./datasets/metas}"
NORM_STATS_PATH="./norm_stats/libero_norm.json"
TRAIN_METAS_PATH="./datasets/metas/libero_train.json"

# SmolVLM backbone (local path from paths.env, else HuggingFace repo)
SMOLVLM_MODEL="${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}"

# =============================================================================
# Training hyperparameters
# =============================================================================
LEARNING_RATE=2e-4
NUM_ACTIONS=10          # Action horizon
ITERS=200000
WARMUP_STEPS=0
FREEZE_STEPS=1000
SAVE_INTERVAL=10000
LOG_INTERVAL=20
NUM_WORKERS=4
MAX_GRAD_NORM=1.0

# Model architecture (Large configuration)
HIDDEN_SIZE=1024
DEPTH=24
NUM_HEADS=16
USE_ADALN=false          # DiT-style conditioning
USE_ADAPTIVE_CHUNKING=${USE_ADAPTIVE_CHUNKING:-false}  # change-rate-weighted loss + boundary head
CHUNK_LOSS_WEIGHT=${CHUNK_LOSS_WEIGHT:-0.1}            # weight of boundary auxiliary loss
TDS_WEIGHTS_CSV=${TDS_WEIGHTS_CSV:-""}                 # task_difficulty.csv -> enables TDS sampling

# =============================================================================
# Step 1: Create training metadata (if not exists)
# =============================================================================
if [ ! -f "$TRAIN_METAS_PATH" ]; then
    echo "Creating training metadata..."
    python create_libero_meta.py \
        --data_dir $LIBERO_DATA_DIR \
        --subsets libero_10 libero_goal libero_object libero_spatial libero_90 \
        --output $TRAIN_METAS_PATH
fi

# =============================================================================
# Step 2: Compute normalization statistics (if not exists)
# =============================================================================
if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "Computing normalization statistics..."
    python compute_libero_norm_stats.py \
        --data_dir $LIBERO_DATA_DIR \
        --subsets libero_10 libero_goal libero_object libero_spatial libero_90 \
        --output $NORM_STATS_PATH
fi

# =============================================================================
# Step 3: Build training arguments
# =============================================================================
ARGS="--output_dir ${OUTPUT_DIR} \
    --train_metas_path ${TRAIN_METAS_PATH} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode libero_joint \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --learning_coef ${LEARNING_COEF} \
    --num_actions ${NUM_ACTIONS} \
    --iters ${ITERS} \
    --warmup_steps ${WARMUP_STEPS} \
    --freeze_steps ${FREEZE_STEPS} \
    --hidden_size ${HIDDEN_SIZE} \
    --depth ${DEPTH} \
    --num_heads ${NUM_HEADS} \
    --num_workers ${NUM_WORKERS} \
    --save_interval ${SAVE_INTERVAL} \
    --log_interval ${LOG_INTERVAL} \
    --image_size 384 \
    --norm_stats_path ${NORM_STATS_PATH} \
    --max_grad_norm ${MAX_GRAD_NORM}"

# Add AdaLN flag if enabled
if [ "${USE_ADALN}" = true ]; then
    ARGS="${ARGS} --use_adaln"
fi

# Add adaptive action chunking if enabled
if [ "${USE_ADAPTIVE_CHUNKING}" = true ]; then
    ARGS="${ARGS} --use_adaptive_chunking --chunk_loss_weight ${CHUNK_LOSS_WEIGHT}"
    echo "Adaptive action chunking ENABLED (chunk_loss_weight=${CHUNK_LOSS_WEIGHT})"
fi

# Add Transition-Density Sampling if a weights csv is given
if [ -n "${TDS_WEIGHTS_CSV}" ]; then
    ARGS="${ARGS} --tds_weights_csv ${TDS_WEIGHTS_CSV}"
    echo "Transition-Density Sampling ENABLED (${TDS_WEIGHTS_CSV})"
fi

# Add resume checkpoint if specified
if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
    echo "Resuming from ${RESUME_CKPT}"
fi

# =============================================================================
# Step 4: Start training
# =============================================================================
echo "============================================================"
echo "Starting SimVLA Training on LIBERO (Large Action Transformer)"
echo "============================================================"
echo "SmolVLM backbone: ${SMOLVLM_MODEL}"
echo "Data directory: $LIBERO_DATA_DIR"
echo "Normalization stats: $NORM_STATS_PATH"
echo "Action mode: libero_joint"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Learning coef: ${LEARNING_COEF}"
echo "Num actions: ${NUM_ACTIONS}"
echo "Image size: 384x384"
echo "============================================================"
echo "Action Transformer configuration:"
echo "   Hidden size: ${HIDDEN_SIZE}"
echo "   Depth: ${DEPTH}"
echo "   Num heads: ${NUM_HEADS}"
echo "   Use AdaLN: ${USE_ADALN}"
echo "   Estimated parameters: ~302M"
echo "============================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"

# Save a full copy of stdout/stderr to a timestamped log for later review
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train_console_$(date +%Y%m%d-%H%M%S).log"
echo "Console log: ${LOG_FILE}"

# Training (num_processes driven by NUM_GPUS from paths.env)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=${NUM_PROCESSES} \
    --main_process_port 29504 \
    --mixed_precision bf16 \
    train_smolvlm.py ${ARGS} 2>&1 | tee "${LOG_FILE}"

echo "Training completed!"
