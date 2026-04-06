#!/bin/bash
# MiroEval — unified three-dimensional evaluation wrapper.
#
# Usage:
#   bash run_eval.sh --input data/method_results/my_model.json --model_name my_model
#   bash run_eval.sh --input results.json --model_name test --evaluations factual_eval point_quality
#
# If the venv doesn't exist, it will be created automatically via `uv sync`.

set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv/bin/python"

# Auto-create venv if missing
if [ ! -x "$VENV" ]; then
    echo "[MiroEval] Virtual env not found. Running uv sync ..."
    uv sync
    echo "[MiroEval] Environment ready."
fi

exec "$VENV" run_eval.py "$@"
