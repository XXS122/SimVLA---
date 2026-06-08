#!/usr/bin/env bash
# Single-task LIBERO experiment runner for inference-time innovations.
#
# Workflow (run the SERVER in one terminal, this CLIENT in another):
#
#   The metric we care about is printed at the end of the client run:
#     - "Total success rate"      -> must stay >= baseline
#     - "Avg physical steps ..."  -> the ~120-step number we want to lower
#
# ---------------------------------------------------------------------------
# STEP 1. Start the server (separate terminal) with the preset you want:
#
#   # baseline  (Euler, NFE=10, no boundary) — reproduces current behaviour
#   CUDA_VISIBLE_DEVICES=0 python serve_smolvlm_libero.py \
#       --checkpoint $SIMVLA_CHECKPOINTS --norm_stats ../../norm_stats/libero_norm.json \
#       --port 8102 --solver euler --nfe_steps 10
#
#   # heun     (2nd-order, NFE=5) — latency win, quality should match baseline
#   CUDA_VISIBLE_DEVICES=0 python serve_smolvlm_libero.py \
#       --checkpoint $SIMVLA_CHECKPOINTS --norm_stats ../../norm_stats/libero_norm.json \
#       --port 8102 --solver heun --nfe_steps 5
#
#   # heun+aac (also emit boundary scores for adaptive chunking on the client)
#   CUDA_VISIBLE_DEVICES=0 python serve_smolvlm_libero.py \
#       --checkpoint $SIMVLA_CHECKPOINTS --norm_stats ../../norm_stats/libero_norm.json \
#       --port 8102 --solver heun --nfe_steps 5 --send_boundary
#
# ---------------------------------------------------------------------------
# STEP 2. Run this script (client). Usage:
#
#   bash run_single_task.sh <preset> [port] [task_id] [num_trials] [gpu]
#     preset      : baseline | heun | aac   (must match the server you started)
#     port        : default 8102
#     task_id     : libero_goal task index, default 0
#     num_trials  : episodes for that task, default 20
#     gpu         : CUDA device for the env renderer, default 0
#
# Examples:
#   bash run_single_task.sh baseline 8102 0 20 0
#   bash run_single_task.sh heun     8102 0 20 0
#   bash run_single_task.sh aac      8102 0 20 0
set -euo pipefail

PRESET="${1:-baseline}"
PORT="${2:-8102}"
TASK_ID="${3:-0}"
NUM_TRIALS="${4:-20}"
GPU="${5:-0}"
TASK_SUITE="libero_goal"

EXTRA=()
case "$PRESET" in
  baseline|heun)
    : ;;  # fixed replan; differences are server-side only
  aac)
    EXTRA+=(--adaptive_chunking --boundary_threshold 0.5 --min_replan 2)
    ;;
  *)
    echo "Unknown preset '$PRESET' (expected: baseline | heun | aac)" >&2
    exit 1 ;;
esac

echo "== Preset: $PRESET | suite: $TASK_SUITE | task_id: $TASK_ID | trials: $NUM_TRIALS =="

CUDA_VISIBLE_DEVICES="$GPU" python libero_client.py \
  --host 127.0.0.1 \
  --port "$PORT" \
  --client_type websocket \
  --task_suite "$TASK_SUITE" \
  --task_id "$TASK_ID" \
  --num_trials "$NUM_TRIALS" \
  --no_video \
  "${EXTRA[@]}"
