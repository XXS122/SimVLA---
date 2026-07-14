#!/bin/bash
# =============================================================================
# SimVLA LIBERO-PRO robustness evaluation (perturbed suites, 4-way parallel)
#
# LIBERO-PRO (github.com/Zxy-MLlab/LIBERO-PRO) extends LIBERO with perturbed
# variants of the four suites, one registered suite per perturbation
# dimension, e.g. libero_goal_lan (semantic), libero_goal_object (object),
# libero_goal_swap (position), libero_goal_task, libero_goal_env. Selecting a
# dimension therefore just means selecting suite names -- this script maps
# <dimension> to the right suffix automatically. The *_temp suites are the
# position-intensity workflow (copy the x0.1..x0.5 / y0.1..y0.5 bddl/init
# variants into the *_temp folders first, then run with dimension "temp").
#
# One-time setup on the eval machine:
#   1) git clone https://github.com/Zxy-MLlab/LIBERO-PRO
#   2) download bddl_files/init_files from the official HuggingFace dataset
#      (huggingface.co/datasets/zhouxueyang/LIBERO-Pro) and move them into
#      LIBERO-PRO's libero/libero/{bddl_files,init_files}
#   3) export LIBERO_PRO_ROOT=/abs/path/to/LIBERO-PRO   (add to paths.env)
#      (do NOT pip install -e it -- PYTHONPATH shadowing below keeps your
#       normal LIBERO install untouched)
#
# Usage:
#   bash run_eval_pro.sh <port> <num_trials> <output_prefix> <dimension> "<gpu1> <gpu2> <gpu3> <gpu4>"
#   <dimension>: semantic|object|position|task|environment|temp
# Example (uniform 100k ckpt served on 8102, single GPU, object perturbation):
#   bash run_eval_pro.sh 8102 20 uni100k object "0 0 0 0"
#
# Outputs (like run_eval_all.sh):
#   <prefix>_<dim>_{spatial,object,goal,10}.txt
#   <prefix>_<dim>_per_task_*.csv  and merged  <prefix>_<dim>_sr_all.csv
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$LIBERO_PRO_ROOT" ] || [ ! -d "$LIBERO_PRO_ROOT" ]; then
    echo "ERROR: LIBERO_PRO_ROOT is unset or not a directory."
    echo "   git clone https://github.com/Zxy-MLlab/LIBERO-PRO"
    echo "   export LIBERO_PRO_ROOT=/abs/path/to/LIBERO-PRO"
    exit 1
fi

# LIBERO-PRO's libero package must shadow the bundled LIBERO so that
# benchmark.get_benchmark_dict() exposes the perturbed suites.
export LIBERO_ROOT="$LIBERO_PRO_ROOT"
export PYTHONPATH="${LIBERO_PRO_ROOT}:${PYTHONPATH}"

# Headless MuJoCo rendering (same as run_eval_all.sh)
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}

PORT=${1:-8102}
NUM_TRIALS=${2:-20}
OUTPUT_PREFIX=${3:-"eval_pro"}
DIM=${4:-"object"}            # semantic|object|position|task|environment|temp
GPUS=${5:-"0 0 0 0"}

read -ra GPU_ARRAY <<< "$GPUS"
if [ ${#GPU_ARRAY[@]} -lt 4 ]; then
    echo "ERROR: need 4 gpu entries (may repeat one id), got ${#GPU_ARRAY[@]}"
    exit 1
fi

# Map dimension -> LIBERO-PRO suite-name suffix
case "$DIM" in
    semantic|language|lan)  SUFFIX="lan" ;;
    object|obj)             SUFFIX="object" ;;
    position|swap|pos)      SUFFIX="swap" ;;
    task)                   SUFFIX="task" ;;
    environment|env)        SUFFIX="env" ;;
    temp)                   SUFFIX="temp" ;;   # position-intensity workflow
    *) echo "ERROR: unknown dimension '$DIM' (use semantic|object|position|task|environment|temp)"; exit 1 ;;
esac

# Suite names; override if your LIBERO-PRO version registers different ones
# (a wrong name makes libero_client.py print the full registered list):
#   PRO_SUITES="a b c d" bash run_eval_pro.sh ...
PRO_SUITES=${PRO_SUITES:-"libero_spatial_${SUFFIX} libero_object_${SUFFIX} libero_goal_${SUFFIX} libero_10_${SUFFIX}"}
read -ra SUITE_ARRAY <<< "$PRO_SUITES"

PREFIX="${OUTPUT_PREFIX}_${DIM}"
OUTPUT_DIR="./eval_pro_${PORT}"
mkdir -p "$OUTPUT_DIR"
rm -f "${PREFIX}"_per_task_*.csv "${PREFIX}_sr_all.csv"

echo "LIBERO-PRO evaluation"
echo "   LIBERO_PRO_ROOT: $LIBERO_PRO_ROOT"
echo "   dimension: $DIM (suite suffix: _${SUFFIX})"
echo "   suites: ${SUITE_ARRAY[*]}"
echo "   port=$PORT trials/task=$NUM_TRIALS prefix=$PREFIX gpus=$GPUS"
echo ""

PIDS=()
for i in 0 1 2 3; do
    suite=${SUITE_ARRAY[$i]}
    short=${suite#libero_}; short=${short%_${SUFFIX}}
    CUDA_VISIBLE_DEVICES=${GPU_ARRAY[$i]} python -u libero_client.py \
        --host 127.0.0.1 \
        --port $PORT \
        --client_type websocket \
        --task_suite $suite \
        --num_trials $NUM_TRIALS \
        --per_task_csv "${PREFIX}_per_task_${short}.csv" \
        --video_out "$OUTPUT_DIR" --no_video > "${PREFIX}_${short}.txt" 2>&1 &
    PIDS+=($!)
    echo "   [PID ${PIDS[-1]}] $suite (GPU ${GPU_ARRAY[$i]}) -> ${PREFIX}_${short}.txt"
done

echo ""
echo "Waiting... (tail -f ${PREFIX}_*.txt to monitor)"
wait "${PIDS[@]}"

echo ""
echo "Results summary ($DIM):"
echo "=========================================="
for suite in "${SUITE_ARRAY[@]}"; do
    short=${suite#libero_}; short=${short%_${SUFFIX}}
    file="${PREFIX}_${short}.txt"
    if [ -f "$file" ]; then
        echo "--- $short ---"
        grep -iE "Total success rate" "$file" 2>/dev/null || echo "  (see $file)"
    fi
done
echo "=========================================="

MERGED="${PREFIX}_sr_all.csv"
awk 'FNR==1 && NR!=1 {next} {print}' "${PREFIX}"_per_task_*.csv > "$MERGED" 2>/dev/null \
    && echo "Per-task SR merged into: $MERGED" \
    || echo "(no per-task csvs found to merge)"
