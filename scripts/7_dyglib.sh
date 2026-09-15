#!/usr/bin/env bash
# Native DyGLib CLI, including --model_name FNN. Run setup and prepare first.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DYGLIB_DIR="${DYGLIB_DIR:-$project_root/derived/dyglib}"
PYTHON="${PYTHON:-$project_root/venv/bin/python}"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$DYGLIB_DIR"
entry=train_link_prediction.py
if [[ "${1:-}" == "eval" ]]; then
  entry=evaluate_link_prediction.py
  shift
fi
exec "$PYTHON" -u "$entry" "$@"
