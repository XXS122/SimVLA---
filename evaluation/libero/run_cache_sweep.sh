#!/usr/bin/env bash
# Dual-rate VLM-caching sweep for a SINGLE LIBERO task (default: libero_goal task 9).
#
# Goal: find the largest VLM refresh interval N that keeps success rate at the
# baseline while cutting average inference latency.
#
# The server is started ONCE (it reads vlm_refresh_every per request), and this
# script runs the client several times with different N. For each run it prints:
#   - Total success rate        -> must stay >= baseline (N=1)
#   - Avg inference latency      -> should drop as N grows
#   - VLM refresh rate           -> fraction of queries that ran the full VLM
#
# ---------------------------------------------------------------------------
# STEP 1 (separate terminal). Start ONE server. refresh_every is overridden
# per request by this script, so any default is fine:
#
#   source paths.env
#   CUDA_VISIBLE_DEVICES=6 python serve_smolvlm_libero.py \
#       --checkpoint "$SIMVLA_CHECKPOINTS" \
#       --norm_stats ../../norm_stats/libero_norm.json \
#       --smolvlm_model "$SIMVLA_SMOLVLM_MODEL" \
#       --port 8102 --solver euler --nfe_steps 10
#
# Wait for "SimVLA server listening on 0.0.0.0:8102".
#
# ---------------------------------------------------------------------------
# STEP 2. Run this script (client side):
#
#   bash run_cache_sweep.sh <port> <task_id> <num_trials> <gpu> [N values...]
#     port        : default 8102
#     task_id     : libero_goal task index, default 9
#     num_trials  : episodes per N, default 20
#     gpu         : CUDA device for the env renderer, default 0
#     N values    : refresh intervals to sweep, default "1 2 3 5"
#
# Examples:
#   bash run_cache_sweep.sh 8102 9 20 0
#   bash run_cache_sweep.sh 8102 9 30 0 "1 2 4"
set -euo pipefail

PORT="${1:-8102}"
TASK_ID="${2:-9}"
NUM_TRIALS="${3:-20}"
GPU="${4:-0}"
N_VALUES="${5:-1 2 3 5}"
TASK_SUITE="libero_goal"
SEED="${SEED:-7}"

OUTDIR="cache_sweep_${TASK_SUITE}_task${TASK_ID}"
mkdir -p "$OUTDIR"

echo "================================================================"
echo " VLM-cache sweep | suite=$TASK_SUITE task_id=$TASK_ID trials=$NUM_TRIALS"
echo " N values: $N_VALUES   (N=1 is the baseline)"
echo " logs -> $OUTDIR/"
echo "================================================================"

for N in $N_VALUES; do
  LOG="$OUTDIR/refresh_${N}.log"
  echo ""
  echo ">>> Running N=$N  (refresh VLM every $N queries) -> $LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python libero_client.py \
    --host 127.0.0.1 \
    --port "$PORT" \
    --client_type websocket \
    --task_suite "$TASK_SUITE" \
    --task_id "$TASK_ID" \
    --num_trials "$NUM_TRIALS" \
    --seed "$SEED" \
    --vlm_refresh_every "$N" \
    --no_video 2>&1 | tee "$LOG"
done

echo ""
echo "================================================================"
echo " SUMMARY (success rate must hold; latency should fall as N grows)"
echo "================================================================"
for N in $N_VALUES; do
  LOG="$OUTDIR/refresh_${N}.log"
  SR=$(grep -E "Total success rate" "$LOG" | tail -1 || true)
  LAT=$(grep -E "Avg inference latency" "$LOG" | tail -1 || true)
  RR=$(grep -E "VLM refresh rate" "$LOG" | tail -1 || true)
  STEPS=$(grep -E "Avg physical steps" "$LOG" | tail -1 || true)
  printf "N=%-3s | %s | %s | %s | %s\n" "$N" "$SR" "$LAT" "$STEPS" "$RR"
done
