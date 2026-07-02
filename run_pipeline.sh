#!/bin/bash
# Thin wrapper around the latent_action pipeline.
#
#   source paths.env            # or let this script source it
#   ./run_pipeline.sh <stage> [options]
#
# Stages: meta | norm-stats | splits | baseline | train-lam | label |
#         probe | pretrain | finetune | serve
# See latent_action/README.md for the full recipe.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment if paths.env exists and key vars are missing
if [ -z "$SIMVLA_CHECKPOINTS" ] && [ -f "$SCRIPT_DIR/paths.env" ]; then
    echo "sourcing paths.env"
    source "$SCRIPT_DIR/paths.env"
fi

export TF_CPP_MIN_LOG_LEVEL=2

exec python -m latent_action.run "$@"
