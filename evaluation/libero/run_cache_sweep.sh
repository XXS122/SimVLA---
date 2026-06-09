#!/usr/bin/env bash
# Adaptive dual-rate VLM-caching sweep for a SINGLE LIBERO task.
#
# Three modes tested:
#   baseline  : VLM runs every query (ground truth)
#   bnd0.5_N2 : conservative — max 1 cached step, boundary gate 0.5
#               Expected ~25% cache, ~8% speedup. Safety-first.
#   bnd0.5_N4 : aggressive  — max 3 cached steps, boundary gate 0.5
#               Expected ~50% cache, ~15% speedup. May trade 5% success.
#
# Together these three points show:
#   1. Naive fixed-N caching collapses to 0% success (included for reference)
#   2. Conservative adaptive caching preserves success rate
#   3. Aggressive adaptive caching reveals the speed-accuracy Pareto frontier
#
# Usage:
#   bash run_cache_sweep.sh [port] [task_id] [num_trials] [gpu]
#     port       : default 8102
#     task_id    : libero_goal task index, default 9
#     num_trials : episodes per mode, default 20
#     gpu        : CUDA device for env renderer, default 0
set -euo pipefail

PORT="${1:-8102}"
TASK_ID="${2:-9}"
NUM_TRIALS="${3:-20}"
GPU="${4:-0}"
TASK_SUITE="libero_goal"
SEED="${SEED:-7}"

OUTDIR="cache_sweep_${TASK_SUITE}_task${TASK_ID}"
mkdir -p "$OUTDIR"

echo "================================================================"
echo " Adaptive VLM-caching sweep"
echo " suite=$TASK_SUITE  task_id=$TASK_ID  trials=$NUM_TRIALS"
echo " logs -> $OUTDIR/"
echo "================================================================"

run_one() {
  local TAG="$1"; shift
  local LOG="$OUTDIR/${TAG}.log"
  echo ""
  echo ">>> $TAG -> $LOG"
  CUDA_VISIBLE_DEVICES="$GPU" python libero_client.py \
    --host 127.0.0.1 --port "$PORT" --client_type websocket \
    --task_suite "$TASK_SUITE" --task_id "$TASK_ID" \
    --num_trials "$NUM_TRIALS" --seed "$SEED" \
    --no_video "$@" 2>&1 | tee "$LOG"
}

# Baseline: VLM runs every query (exact original behaviour)
run_one "baseline_N1" \
  --vlm_refresh_every 1

# Conservative: max 1 cached step (N=2), boundary gate 0.5.
# Boundary score is bimodal: ~0 during smooth motion, ~0.87 at contact.
# N=2 means at most 1 consecutive cached query = 5 physical steps on same
# visual features. This is the safety-first operating point.
run_one "bnd0.5_N2" \
  --vlm_refresh_every 2 --boundary_refresh_thresh 0.5

# Aggressive: max 3 cached steps (N=4), same boundary gate.
# Prior experiments: 34/40 (85%) vs baseline 42/45 (93%) — Pareto point.
run_one "bnd0.5_N4" \
  --vlm_refresh_every 4 --boundary_refresh_thresh 0.5

echo ""
echo "================================================================"
echo " SUMMARY"
echo " Key: success rate must hold vs baseline; latency should fall"
echo "================================================================"
printf "%-20s | %-30s | %-38s | %-25s | %s\n" \
  "MODE" "SUCCESS RATE" "AVG LATENCY" "AVG STEPS" "VLM REFRESH RATE"
echo "---"
for TAG in baseline_N1 bnd0.5_N2 bnd0.5_N4; do
  LOG="$OUTDIR/${TAG}.log"
  SR=$(grep -E "^Total success rate" "$LOG" | tail -1 || echo "?")
  LAT=$(grep -E "^Avg inference latency" "$LOG" | tail -1 || echo "?")
  STEPS=$(grep -E "^Avg physical steps" "$LOG" | tail -1 || echo "?")
  RR=$(grep -E "^VLM refresh rate" "$LOG" | tail -1 || echo "?")
  printf "%-20s | %-30s | %-38s | %-25s | %s\n" \
    "$TAG" "$SR" "$LAT" "$STEPS" "$RR"
done
