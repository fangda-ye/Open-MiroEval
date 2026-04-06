#!/bin/bash
# MiroEval — shell wrapper for the unified CLI.
#
# Usage:
#   bash scripts/run_eval.sh run --input data/method_results/qwen3.json --model qwen3
#   bash scripts/run_eval.sh status
#   bash scripts/run_eval.sh retry --model qwen3

set -euo pipefail
cd "$(dirname "$0")/.."

# Use conda env if available, otherwise fall back to system python.
if command -v conda &>/dev/null && conda env list | grep -q miroeval; then
    exec conda run -n miroeval --no-capture-output python -m miroeval "$@"
else
    VENV=".venv/bin/python"
    if [ ! -x "$VENV" ]; then
        echo "[MiroEval] No conda env 'miroeval' or .venv found. Run: conda create -n miroeval python=3.11"
        exit 1
    fi
    exec "$VENV" -m miroeval "$@"
fi
