#!/usr/bin/env bash
# Adaptive dual-rate VLM-caching sweep for a SINGLE LIBERO task.
#
# Three trigger modes are tested:
#   N=1           : baseline (no caching, VLM runs every query)
#   N=5           : naive fixed-N caching (fast, but may drop success rate)
#   N=5+boundary  : adaptive — boundary head fires at contact/reversal moments,
#                   so the VLM refreshes semantically rather than blindly.
#                   This is the core contribution: same or better success rate
#                   as baseline with average latency close to N=5.
#
# ---------------------------------------------------------------------------
# STEP 1 (separate terminal). Start ONE server — it reads all cache knobs
# per request, so this single instance handles all three modes:
#
#   source paths.env
#   CUDA_VISIBLE_DEVICES=6 python serve_smolvlm_libero.py \
#       --checkpoint "$SIMVLA_CHECKPOINTS" \
#       --norm_stats ../../norm_stats/libero_norm.json \
#       --smolvlm_model "$SIMVLA_SMOLVLM_MODEL" \
#       --port 8102 --solver euler --nfe_steps 10
#
# ---------------------------------------------------------------------------
# STEP 2. Run this script:
#
#   bash run_cache_sweep.sh [port] [task_id] [num_trials] [gpu]
#     port       : default 8102
#     task_id    : libero_goal task index, default 9
#     num_trials : episodes per mode, default 20
#     gpu        : CUDA device for env renderer, default 0
#
# Examples:
#   bash run_cache_sweep.sh 8102 9 5 6    # quick 5-ep sanity check on task 9
#   bash run_cache_sweep.sh 8102 9 20 6   # full 20-ep comparison
set -euo pipefail

PORT="${1:-8102}"
TASK_ID="${2:-9}"
NUM_TRIALS="${3:-10}"
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

# Baseline: VLM runs every query (ground-truth success rate)
run_one "baseline_N1" \
  --vlm_refresh_every 1

# Boundary-only gate (calibrated thresh 0.5; boundary score is bimodal:
# ~0 during smooth motion, ~0.9 at contact). Caches ~50% but the boundary
# head can't sense free-space drift from stale vision -> can drop success.
run_one "bnd0.5_N4" \
  --vlm_refresh_every 4 --boundary_refresh_thresh 0.5

# Boundary + image-change gate (the safe combination): cache only when the
# scene is nearly static AND no contact predicted. img_thresh 0.2 ~ median
# frame RMS, so any appreciable visual motion forces a refresh.
run_one "bnd0.5_img0.2_N4" \
  --vlm_refresh_every 4 --boundary_refresh_thresh 0.5 --vlm_img_thresh 0.2

# Image-only gate (training-free, checkpoint-agnostic): refresh purely on
# visual change. Most principled "do I need to recompute vision" signal.
run_one "img0.15_N6" \
  --vlm_refresh_every 6 --vlm_img_thresh 0.15

echo ""
echo "================================================================"
echo " SUMMARY"
echo " Key: success rate must hold vs baseline; latency should fall"
echo "================================================================"
printf "%-28s | %-30s | %-35s | %-30s | %s\n" \
  "MODE" "SUCCESS RATE" "AVG LATENCY" "AVG STEPS" "VLM REFRESH RATE"
echo "---"
for TAG in baseline_N1 bnd0.5_N4 bnd0.5_img0.2_N4 img0.15_N6; do
  LOG="$OUTDIR/${TAG}.log"
  SR=$(grep -E "^Total success rate" "$LOG" | tail -1 || echo "?")
  LAT=$(grep -E "^Avg inference latency" "$LOG" | tail -1 || echo "?")
  STEPS=$(grep -E "^Avg physical steps" "$LOG" | tail -1 || echo "?")
  RR=$(grep -E "^VLM refresh rate" "$LOG" | tail -1 || echo "?")
  printf "%-28s | %-30s | %-35s | %-30s | %s\n" \
    "$TAG" "$SR" "$LAT" "$STEPS" "$RR"
done
