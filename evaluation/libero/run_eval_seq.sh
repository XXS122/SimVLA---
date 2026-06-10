#!/bin/bash
# =============================================================================
# SimVLA LIBERO Evaluation Script — sequential, single GPU
#
# Same outputs as run_eval_all.sh (per-suite txt logs, per-task SR csvs,
# merged <prefix>_sr_all.csv for transition_density_stats.py --sr_csv),
# but runs the 4 task suites one after another on a single GPU.
#
# Usage:
#   bash run_eval_seq.sh <port> <num_trials> <output_prefix> <gpu> [--no_video]
# Example:
#   bash run_eval_seq.sh 8102 20 baseline_exp0 0 --no_video
# =============================================================================

set -e

# LIBERO environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIBERO_ROOT="${SCRIPT_DIR}/LIBERO"
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"

PORT=${1:-8102}
NUM_TRIALS=${2:-20}
OUTPUT_PREFIX=${3:-"eval_simvla"}
GPU=${4:-0}
EXTRA_ARGS=${5:-}   # e.g. --no_video

OUTPUT_DIR="./eval_simvla_${PORT}"
mkdir -p "$OUTPUT_DIR"

# libero_client.py appends to per-task csvs: clear leftovers from previous
# runs with the same prefix
rm -f "${OUTPUT_PREFIX}"_per_task_*.csv "${OUTPUT_PREFIX}_sr_all.csv"

echo "Sequential LIBERO evaluation (single GPU $GPU, port $PORT, $NUM_TRIALS trials/task)"
echo ""

for suite in libero_spatial libero_object libero_goal libero_10; do
    short=${suite#libero_}
    echo ">>> [$suite] -> ${OUTPUT_PREFIX}_${short}.txt"
    CUDA_VISIBLE_DEVICES=$GPU python -u libero_client.py \
        --host 127.0.0.1 \
        --port $PORT \
        --client_type websocket \
        --task_suite $suite \
        --num_trials $NUM_TRIALS \
        --per_task_csv "${OUTPUT_PREFIX}_per_task_${short}.csv" \
        --video_out "$OUTPUT_DIR" $EXTRA_ARGS 2>&1 | tee "${OUTPUT_PREFIX}_${short}.txt" \
        | grep -E "Task suite|Total success rate" || true
done

echo ""
echo "Results summary:"
echo "=========================================="
for suite in spatial object goal 10; do
    file="${OUTPUT_PREFIX}_${suite}.txt"
    if [ -f "$file" ]; then
        echo "--- $suite ---"
        grep -E "Total success rate" "$file" 2>/dev/null || echo "  (see $file)"
    fi
done
echo "=========================================="

# Merge per-suite SR csvs (name must NOT match the _per_task_*.csv glob,
# or a rerun would merge it into itself)
MERGED="${OUTPUT_PREFIX}_sr_all.csv"
awk 'FNR==1 && NR!=1 {next} {print}' "${OUTPUT_PREFIX}"_per_task_*.csv > "$MERGED" 2>/dev/null \
    && echo "Per-task SR merged into: $MERGED" \
    || echo "(no per-task csvs found to merge)"
