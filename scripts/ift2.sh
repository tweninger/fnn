#!/usr/bin/env bash
set -euo pipefail

PYTHON=venv/bin/python
RESULTS_DIR=derived/results
LOG_DIR=derived/logs
GPU_ID="${GPU_ID:-0}"

TOPOLOGIES=(ring grid torus doorway swisscheese)
SEEDS=(0 1 2 3 4)
TRAIN_ROLL_STEPS=(1 4 7 10 13 16 19)

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

for seed in "${SEEDS[@]}"; do
  for topology in "${TOPOLOGIES[@]}"; do
    for train_steps in "${TRAIN_ROLL_STEPS[@]}"; do
      stem="second_order_wave_${topology}_seed${seed}_trainroll${train_steps}_cutoff5"
      result="$RESULTS_DIR/${stem}.jsonl"
      log="$LOG_DIR/${stem}.log"

      if [[ -e "$result" ]]; then
        echo "Skipping existing result: $result"
        continue
      fi

      echo "=== topology=$topology seed=$seed ==="

      CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -m interactiondynamics.train quick \
        --dataset synthetic \
        --synthetic-task wave \
        --synthetic-topology "$topology" \
        --synthetic-num-nodes 64 \
        --seed "$seed" \
        --ift-variants auto \
        --ift-orders 2 \
	--ift-history-steps 1 2 3 \
        --epochs 10 \
        --num-bins 192 \
        --rollout-horizon 20 \
	--rollout-train-steps "$train_steps" \
        --synthetic-drive-cutoff 5 \
        --save-jsonl "$result" \
        2>&1 | tee "$log"
    done
  done
done
