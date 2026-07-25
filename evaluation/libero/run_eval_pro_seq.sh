#!/bin/bash
# =============================================================================
# Run all four LIBERO-PRO perturbation dimensions back-to-back against the
# checkpoint currently served on <port>, printing a per-dimension average SR.
#
# It wraps run_eval_pro.sh: object -> position -> semantic -> task, one after
# another (a failed dimension is reported but does not stop the rest). After
# each dimension it parses the four suites' "Total success rate" and prints
# their mean (= the paper's Avg). A summary table is written at the end.
#
# The checkpoint is whatever the server on <port> loaded, so the two budget
# points you want are two separate runs:
#   1) serve ckpt-10000 on 8102, then:
#        bash run_eval_pro_seq.sh 8102 20 u10k "0 0 0 0"
#   2) re-serve ckpt-40000 on 8102, then:
#        bash run_eval_pro_seq.sh 8102 20 u40k "0 0 0 0"
#   (for the TDS model, serve its checkpoints and use prefixes tds10k / tds40k)
#
# Usage:
#   bash run_eval_pro_seq.sh <port> <num_trials> <prefix> "<gpu1> <gpu2> <gpu3> <gpu4>"
# Override the dimension list if needed:
#   DIMS="object task" bash run_eval_pro_seq.sh 8102 20 u10k "0 0 0 0"
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT=${1:-8102}
NUM_TRIALS=${2:-20}
PREFIX=${3:-eval}
GPUS=${4:-"0 0 0 0"}
DIMS=${DIMS:-"object position semantic task"}

# Extract the "(NN.N%)" percentage from a suite .txt's "Total success rate" line
pct() {
    grep -i "Total success rate" "$1" 2>/dev/null \
        | grep -oE '[0-9]+(\.[0-9]+)?%' | tr -d '%' | tail -1
}

# Mean of the (possibly incomplete) list of numbers passed as arguments
mean() {
    printf '%s\n' "$@" | awk '$1!=""{s+=$1;n++} END{if(n>0) printf "%.1f", s/n; else printf "NA"}'
}

SUMMARY="${PREFIX}_pro_SUMMARY.txt"
HEADER=$(printf "%-10s %8s %8s %8s %8s %8s" "dimension" "Spatial" "Object" "Goal" "Long" "Avg")
echo "$HEADER" > "$SUMMARY"

echo "=========================================================="
echo "LIBERO-PRO sequential evaluation"
echo "   port=$PORT  trials/task=$NUM_TRIALS  prefix=$PREFIX  gpus=$GPUS"
echo "   dimensions: $DIMS"
echo "=========================================================="

for dim in $DIMS; do
    echo ""
    echo "################## dimension: $dim ##################"
    bash "$SCRIPT_DIR/run_eval_pro.sh" "$PORT" "$NUM_TRIALS" "$PREFIX" "$dim" "$GPUS" \
        || echo "[warn] dimension '$dim' returned non-zero; continuing to next"

    sp=$(pct "${PREFIX}_pro_${dim}_spatial.txt")
    ob=$(pct "${PREFIX}_pro_${dim}_object.txt")
    go=$(pct "${PREFIX}_pro_${dim}_goal.txt")
    lo=$(pct "${PREFIX}_pro_${dim}_10.txt")
    avg=$(mean "$sp" "$ob" "$go" "$lo")

    row=$(printf "%-10s %8s %8s %8s %8s %8s" \
          "$dim" "${sp:-NA}" "${ob:-NA}" "${go:-NA}" "${lo:-NA}" "$avg")
    echo "$row" >> "$SUMMARY"
    echo ""
    echo ">>> [$dim] per-suite = ${sp:-NA}/${ob:-NA}/${go:-NA}/${lo:-NA}  ->  average = ${avg}%"
done

echo ""
echo "=========================================================="
echo "LIBERO-PRO summary  (prefix=$PREFIX)"
echo "=========================================================="
cat "$SUMMARY"
echo ""
echo "(saved to $SUMMARY)"
