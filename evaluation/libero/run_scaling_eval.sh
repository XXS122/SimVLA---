#!/bin/bash
# Task-2: Inference-time scaling sweep (best-of-K) on LIBERO.
#
# For each K in K_LIST, starts the policy server in bok mode, runs the LIBERO
# client, parses the success rate, and appends a CSV row. Also runs the two
# reference points: the 10-step flow baseline and the K=1 one-step generator.
#
# Requires two conda envs on this machine (see evaluation/libero/README.md):
#   $CONDA_SIMVLA (server, default "simvla") and $CONDA_LIBERO (client, "libero").
#
# Usage:
#   source paths.env
#   bash run_scaling_eval.sh <GENERATOR_CKPT> <VERIFIER_CKPT> [SUITE] [TRIALS] [PORT]
set -e

GENERATOR_CKPT=${1:?usage: run_scaling_eval.sh GENERATOR_CKPT VERIFIER_CKPT [SUITE] [TRIALS] [PORT]}
VERIFIER_CKPT=${2:?usage: run_scaling_eval.sh GENERATOR_CKPT VERIFIER_CKPT [SUITE] [TRIALS] [PORT]}
SUITE=${3:-libero_spatial}
TRIALS=${4:-10}
PORT=${5:-8102}

K_LIST=${K_LIST:-"2 4 8 16 32"}
# 10-step flow baseline runs on the SFT teacher when TEACHER_CKPT is set,
# otherwise on the generator itself (multi-step MeanFlow integration).
TEACHER_CKPT=${TEACHER_CKPT:-$GENERATOR_CKPT}
CONDA_SIMVLA=${CONDA_SIMVLA:-simvla}
CONDA_LIBERO=${CONDA_LIBERO:-libero}
NORM_STATS=${NORM_STATS:-../../norm_stats/libero_norm.json}
SMOLVLM_MODEL=${SIMVLA_SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}
GPU=${CUDA_DEVICES:-0}

RESULTS_DIR=${RESULTS_DIR:-./scaling_results}
mkdir -p "$RESULTS_DIR"
CSV="$RESULTS_DIR/scaling_results.csv"
[ -f "$CSV" ] || echo "mode,K,suite,success_rate,episodes,mean_latency_ms" > "$CSV"

wait_for_port() {
    for _ in $(seq 1 120); do
        if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', $1)); s.close()" 2>/dev/null; then
            return 0
        fi
        sleep 2
    done
    echo "Server on port $1 never came up" >&2
    return 1
}

mean_latency() {
    # Mean of latency_ms over a JSONL file (0 if missing/empty)
    python3 - "$1" <<'EOF'
import json, sys
try:
    vals = [json.loads(l)["latency_ms"] for l in open(sys.argv[1]) if l.strip()]
    print(f"{sum(vals)/len(vals):.2f}" if vals else "0")
except FileNotFoundError:
    print("0")
EOF
}

run_one() {
    local MODE=$1 K=$2 CKPT=${3:-$GENERATOR_CKPT}
    local TAG="${MODE}_K${K}_${SUITE}"
    local LAT_LOG="$RESULTS_DIR/latency_${TAG}.jsonl"
    local EVAL_LOG="$RESULTS_DIR/eval_${TAG}.log"
    rm -f "$LAT_LOG"

    echo "=== [$TAG] starting server (mode=$MODE K=$K ckpt=$CKPT) ==="
    local SERVER_ARGS="--checkpoint $CKPT --norm_stats $NORM_STATS \
        --smolvlm_model $SMOLVLM_MODEL --port $PORT --mode $MODE \
        --num_samples $K --latency_log $LAT_LOG"
    if [ "$MODE" = "bok" ]; then
        SERVER_ARGS="$SERVER_ARGS --verifier $VERIFIER_CKPT"
    fi

    CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n "$CONDA_SIMVLA" \
        python serve_smolvlm_libero.py $SERVER_ARGS \
        > "$RESULTS_DIR/server_${TAG}.log" 2>&1 &
    local SERVER_PID=$!
    trap "kill $SERVER_PID 2>/dev/null || true" EXIT
    wait_for_port "$PORT"

    echo "=== [$TAG] running LIBERO client ($TRIALS trials/task) ==="
    conda run --no-capture-output -n "$CONDA_LIBERO" \
        python libero_client.py --port "$PORT" --task_suite "$SUITE" \
        --num_trials "$TRIALS" --no_video 2>&1 | tee "$EVAL_LOG"

    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    trap - EXIT

    # "Total success rate: 42/50 (84.0%)"
    local LINE
    LINE=$(grep "Total success rate" "$EVAL_LOG" | tail -1)
    local SUCC EPS
    SUCC=$(echo "$LINE" | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
    EPS=$(echo "$LINE" | sed -E 's/.*: [0-9]+\/([0-9]+) .*/\1/')
    local SR
    SR=$(python3 -c "print(f'{${SUCC:-0}/100:.4f}')")
    local LAT
    LAT=$(mean_latency "$LAT_LOG")

    echo "$MODE,$K,$SUITE,$SR,${EPS:-0},$LAT" >> "$CSV"
    echo "=== [$TAG] success=$SR latency=${LAT}ms -> $CSV ==="
}

# Reference points: multi-step flow baseline (teacher) + one-step generator (K=1)
run_one flow 1 "$TEACHER_CKPT"
run_one onestep 1

# Best-of-K sweep
for K in $K_LIST; do
    run_one bok "$K"
done

echo ""
echo "Sweep complete. Fit the scaling law with:"
echo "  python fit_scaling_law.py --csv $CSV \\"
echo "      --out_json $RESULTS_DIR/scaling_fit.json --out_plot $RESULTS_DIR/scaling_fit.png"
