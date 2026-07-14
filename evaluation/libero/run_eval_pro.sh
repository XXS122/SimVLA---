#!/bin/bash
# =============================================================================
# SimVLA LIBERO-PRO robustness evaluation (perturbed suites, 4-way parallel)
#
# LIBERO-PRO (github.com/Zxy-MLlab/LIBERO-PRO) extends LIBERO with perturbed
# variants of the four suites, registered as libero_<suite>_temp. The ACTIVE
# perturbation dimension -- object / swap(position) / language(semantic) /
# task / environment -- is configured on the LIBERO-PRO side (its
# evaluation_config.yaml + which bddl/init variant is placed in the *_temp
# folders) BEFORE running this script. The <dimension> argument here is only
# a label for the output files; run this script once per dimension.
#
# One-time setup on the eval machine:
#   1) git clone https://github.com/Zxy-MLlab/LIBERO-PRO
#   2) download its bddl_files / init_files from the official HuggingFace
#      dataset into LIBERO-PRO's libero/libero/{bddl_files,init_files}
#   3) export LIBERO_PRO_ROOT=/abs/path/to/LIBERO-PRO   (add to paths.env)
#
# Usage:
#   bash run_eval_pro.sh <port> <num_trials> <output_prefix> <dimension> "<gpu1> <gpu2> <gpu3> <gpu4>"
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
# benchmark.get_benchmark_dict() exposes the *_temp suites.
export LIBERO_ROOT="$LIBERO_PRO_ROOT"
export PYTHONPATH="${LIBERO_PRO_ROOT}:${PYTHONPATH}"

# Headless MuJoCo rendering (same as run_eval_all.sh)
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}

PORT=${1:-8102}
NUM_TRIALS=${2:-20}
OUTPUT_PREFIX=${3:-"eval_pro"}
DIM=${4:-"object"}            # label only: object|position|semantic|task|environment
GPUS=${5:-"0 0 0 0"}

read -ra GPU_ARRAY <<< "$GPUS"
if [ ${#GPU_ARRAY[@]} -lt 4 ]; then
    echo "ERROR: need 4 gpu entries (may repeat one id), got ${#GPU_ARRAY[@]}"
    exit 1
fi

# LIBERO-PRO perturbed suite names; override if your LIBERO-PRO version
# registers different ones:  PRO_SUITES="a b c d" bash run_eval_pro.sh ...
PRO_SUITES=${PRO_SUITES:-"libero_spatial_temp libero_object_temp libero_goal_temp libero_10_temp"}
read -ra SUITE_ARRAY <<< "$PRO_SUITES"

PREFIX="${OUTPUT_PREFIX}_${DIM}"
OUTPUT_DIR="./eval_pro_${PORT}"
mkdir -p "$OUTPUT_DIR"
rm -f "${PREFIX}"_per_task_*.csv "${PREFIX}_sr_all.csv"

echo "LIBERO-PRO evaluation"
echo "   LIBERO_PRO_ROOT: $LIBERO_PRO_ROOT"
echo "   dimension label: $DIM   (configure the actual perturbation in LIBERO-PRO first!)"
echo "   suites: ${SUITE_ARRAY[*]}"
echo "   port=$PORT trials/task=$NUM_TRIALS prefix=$PREFIX gpus=$GPUS"
echo ""

PIDS=()
for i in 0 1 2 3; do
    suite=${SUITE_ARRAY[$i]}
    short=${suite#libero_}; short=${short%_temp}
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
    short=${suite#libero_}; short=${short%_temp}
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
