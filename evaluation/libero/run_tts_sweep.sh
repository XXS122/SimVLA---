#!/bin/bash
# =============================================================================
# Test-Time Scaling sweep for SimVLA on LIBERO
#
# Runs the (num_samples x ode_steps) grid against ONE running server.
# The server should be started with a diagnostics log, e.g.:
#
#   python serve_smolvlm_libero.py \
#       --checkpoint <ckpt> --norm_stats ../../norm_stats/libero_norm.json \
#       --port 8102 --diag_log ./tts_sweep/diag.jsonl
#
# Usage:
#   bash run_tts_sweep.sh <port> <task_suite> <num_trials> <outdir> [selector]
#
# Grid can be overridden via env vars:
#   N_LIST="1 2 4 8 16" S_LIST="2 5 10 20" bash run_tts_sweep.sh ...
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [ -f "${REPO_ROOT}/paths.env" ]; then
    echo "Sourcing ${REPO_ROOT}/paths.env"
    source "${REPO_ROOT}/paths.env"
fi
export LIBERO_ROOT="${LIBERO_ROOT:-${SCRIPT_DIR}/LIBERO}"
export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH}"

PORT=${1:-8102}
SUITE=${2:-libero_spatial}
NUM_TRIALS=${3:-20}
OUTDIR=${4:-./tts_sweep}
SELECTOR=${5:-consensus}

N_LIST=${N_LIST:-"1 2 4 8 16"}
S_LIST=${S_LIST:-"5 10"}

mkdir -p "$OUTDIR"

echo "TTS sweep: suite=$SUITE trials=$NUM_TRIALS selector=$SELECTOR"
echo "   N grid: $N_LIST"
echo "   S grid: $S_LIST"
echo "   Output: $OUTDIR"
echo ""

for S in $S_LIST; do
  for N in $N_LIST; do
    TAG="N${N}_S${S}_${SELECTOR}"
    RESULTS="$OUTDIR/results_${SUITE}_${TAG}.jsonl"
    if [ -f "$RESULTS" ]; then
      echo ">> Skipping $TAG (results exist: $RESULTS)"
      continue
    fi
    echo ">> Running $TAG ..."
    LOG="$OUTDIR/log_${SUITE}_${TAG}.txt"
    if python -u libero_client.py \
        --host 127.0.0.1 \
        --port "$PORT" \
        --client_type websocket \
        --task_suite "$SUITE" \
        --num_trials "$NUM_TRIALS" \
        --no_video \
        --num_samples "$N" \
        --ode_steps "$S" \
        --selector "$SELECTOR" \
        --log_results "$RESULTS" \
        > "$LOG" 2>&1; then
        tail -1 "$LOG"
    else
        echo "!! FAILED: $TAG — last 30 lines of $LOG:"
        tail -30 "$LOG"
        exit 1
    fi
  done
done

echo ""
echo "Sweep done. Analyze with:"
echo "   python analyze_tts.py --results_dir $OUTDIR --diag $OUTDIR/diag.jsonl"
