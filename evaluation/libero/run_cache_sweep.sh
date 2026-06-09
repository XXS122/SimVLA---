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

# Baseline: VLM runs every query (ground-truth success rate)
run_one "baseline_N1" \
  --vlm_refresh_every 1

# Baseline: VLM runs every query (ground-truth success rate)
run_one "baseline_N1" \
  --vlm_refresh_every 1

# Boundary gate (50% cache in prior runs, ~15% speedup).
# 17/20 vs 18/20 baseline -- statistically indistinguishable at n=20.
run_one "bnd0.5_N4" \
  --vlm_refresh_every 4 --boundary_refresh_thresh 0.5

# Fixed rolling image gate (bug fixed: last_image now updated every step).
# p25 of 5-step frame-RMS = 0.128, so thresh=0.15 targets ~30% cache rate.
# N=3 cap: max 2 consecutive cached steps = 15 physical steps on same vision.
run_one "img0.15_N3_rolling" \
  --vlm_refresh_every 3 --vlm_img_thresh 0.15

# Looser rolling image gate: p50 of 5-step RMS = 0.203, thresh=0.20 targets
# ~50% cache rate (matches boundary gate). N=3 safety cap as above.
run_one "img0.20_N3_rolling" \
  --vlm_refresh_every 3 --vlm_img_thresh 0.20

echo ""
echo "================================================================"
echo " SUMMARY"
echo " Key: success rate must hold vs baseline; latency should fall"
echo "================================================================"
printf "%-28s | %-30s | %-35s | %-30s | %s\n" \
  "MODE" "SUCCESS RATE" "AVG LATENCY" "AVG STEPS" "VLM REFRESH RATE"
echo "---"
for TAG in baseline_N1 bnd0.5_N4 img0.15_N3_rolling img0.20_N3_rolling; do
  LOG="$OUTDIR/${TAG}.log"
  SR=$(grep -E "^Total success rate" "$LOG" | tail -1 || echo "?")
  LAT=$(grep -E "^Avg inference latency" "$LOG" | tail -1 || echo "?")
  STEPS=$(grep -E "^Avg physical steps" "$LOG" | tail -1 || echo "?")
  RR=$(grep -E "^VLM refresh rate" "$LOG" | tail -1 || echo "?")
  printf "%-28s | %-30s | %-35s | %-30s | %s\n" \
    "$TAG" "$SR" "$LAT" "$STEPS" "$RR"
done
