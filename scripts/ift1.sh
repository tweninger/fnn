#!/usr/bin/env bash
set -euo pipefail

PYTHON=venv/bin/python
RESULTS_DIR=derived/results
LOG_DIR=derived/logs
GPU_ID="${GPU_ID:-0}"

TOPOLOGIES=(ring grid torus doorway swisscheese)
SEEDS=(0 1 2 3 4)

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

for topology in "${TOPOLOGIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    result="$RESULTS_DIR/first_order_diffusion_${topology}_seed${seed}_cutoff5.jsonl"
    log="$LOG_DIR/first_order_diffusion_${topology}_seed${seed}_cutoff5.log"

    if [[ -e "$result" ]]; then
      echo "Skipping existing result: $result"
      continue
    fi

    echo "=== topology=$topology seed=$seed ==="

    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -m interactiondynamics.train quick \
      --dataset synthetic \
      --synthetic-task diffusion \
      --synthetic-topology "$topology" \
      --synthetic-num-nodes 64 \
      --seed "$seed" \
      --ift-variants generic \
      --ift-orders 1 \
      --epochs 10 \
      --num-bins 192 \
      --rollout-horizon 20 \
      --synthetic-drive-cutoff 5 \
      --save-jsonl "$result" \
      2>&1 | tee "$log"
  done
done
