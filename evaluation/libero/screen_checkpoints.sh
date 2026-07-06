#!/bin/bash
# =============================================================================
# Screen multiple SimVLA checkpoints against one LIBERO task suite.
#
# For each checkpoint: starts the eval server (simvla conda env), waits for
# it to accept connections, runs a client eval (libero conda env), stops the
# server, then moves to the next checkpoint. Prints a final summary table.
#
# Usage:
#   bash screen_checkpoints.sh <norm_stats_path> <task_suite> <num_trials> \
#       <outdir> <ckpt1> [<ckpt2> ...]
#
# Env overrides (or set in paths.env, which is auto-sourced if present):
#   SIMVLA_ENV   conda env for the server   (default: simvla)
#   LIBERO_ENV   conda env for the client   (default: libero)
#   PORT         websocket port             (default: 8102)
#   CUDA_DEVICES GPU(s) for the server      (default: 0)
#
# If `conda run` is not set up correctly on your machine, run the two halves
# manually instead (one terminal per env) using the same server/client
# commands seen in server_<name>.log / eval_<name>.log after a first attempt.
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${REPO_ROOT}/paths.env" ]; then
    echo "Sourcing ${REPO_ROOT}/paths.env"
    source "${REPO_ROOT}/paths.env"
fi

if [ "$#" -lt 5 ]; then
    echo "Usage: bash screen_checkpoints.sh <norm_stats_path> <task_suite> <num_trials> <outdir> <ckpt1> [<ckpt2> ...]"
    exit 1
fi

NORM_STATS=$1
TASK_SUITE=$2
NUM_TRIALS=$3
OUTDIR=$4
shift 4
CKPTS=("$@")

SIMVLA_ENV="${SIMVLA_ENV:-simvla}"
LIBERO_ENV="${LIBERO_ENV:-libero}"
PORT="${PORT:-8102}"
GPU="${CUDA_DEVICES:-0}"

mkdir -p "$OUTDIR"
declare -a NAMES RATES

wait_for_port() {
    for _ in $(seq 1 90); do
        if (exec 3<>/dev/tcp/127.0.0.1/"$PORT") 2>/dev/null; then
            exec 3<&- 3>&-
            return 0
        fi
        sleep 2
    done
    return 1
}

echo "Screening ${#CKPTS[@]} checkpoint(s) on $TASK_SUITE ($NUM_TRIALS trials/task)"
echo "  server env=$SIMVLA_ENV  client env=$LIBERO_ENV  port=$PORT  gpu=$GPU"
echo ""

for ckpt in "${CKPTS[@]}"; do
    name=$(basename "$ckpt")
    echo "============================================================"
    echo ">> [$name] starting server ..."
    CUDA_VISIBLE_DEVICES="$GPU" conda run -n "$SIMVLA_ENV" python "${SCRIPT_DIR}/serve_smolvlm_libero.py" \
        --checkpoint "$ckpt" --norm_stats "$NORM_STATS" --port "$PORT" \
        > "$OUTDIR/server_${name}.log" 2>&1 &
    SERVER_PID=$!

    if ! wait_for_port; then
        echo "!! [$name] server did not come up within 3 minutes — see $OUTDIR/server_${name}.log"
        kill "$SERVER_PID" 2>/dev/null
        wait "$SERVER_PID" 2>/dev/null
        NAMES+=("$name")
        RATES+=("SERVER_FAILED")
        continue
    fi

    echo ">> [$name] server ready, running eval ..."
    conda run -n "$LIBERO_ENV" python "${SCRIPT_DIR}/libero_client.py" \
        --port "$PORT" --task_suite "$TASK_SUITE" --num_trials "$NUM_TRIALS" --no_video \
        --num_samples 1 --ode_steps 5 \
        --log_results "$OUTDIR/results_${name}.jsonl" \
        > "$OUTDIR/eval_${name}.log" 2>&1

    rate=$(grep "Total success rate" "$OUTDIR/eval_${name}.log" | tail -1 | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    rate="${rate:-N/A}"
    echo ">> [$name] success rate: ${rate}%"
    NAMES+=("$name")
    RATES+=("$rate")

    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    sleep 2
done

echo ""
echo "=== Screening summary ($TASK_SUITE, $NUM_TRIALS trials/task) ==="
for i in "${!NAMES[@]}"; do
    printf "%-30s %s%%\n" "${NAMES[$i]}" "${RATES[$i]}"
done
