#!/usr/bin/env bash
# Reproducible second-order free-response study.
#
# Stages 1 and 4: driven, free/oracle-neighbor, and self-free comparison over
# every dynamic, topology, and seed.  Stage 2: mixed self-free training.  Stage
# 3: differentiable unroll-length ablation.  Stage 5 is derived directly from
# the per-step driven/free/self-free metrics in these JSONLs (no extra training).
set -euo pipefail

PYTHON="${PYTHON:-venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
RESULTS_DIR="${RESULTS_DIR:-derived/results/free_response}"
LOG_DIR="${LOG_DIR:-derived/logs/free_response}"
EPOCHS="${EPOCHS:-20}"
NUM_BINS="${NUM_BINS:-192}"
NUM_NODES="${NUM_NODES:-64}"
HORIZON="${HORIZON:-20}"
CUTOFF="${CUTOFF:-5}"

DYNAMICS=(wave coupled_oscillator)
TOPOLOGIES=(ring grid torus doorway swisscheese)
SEEDS=(0 1 2 3 4)
UNROLLS=(1 5 10)
FREE_TRAIN_PERCENTS=(0 25 50)

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

run_one() {
  local label="$1"
  shift
  local result="$RESULTS_DIR/${label}.jsonl"
  local log="$LOG_DIR/${label}.log"
  if [[ -e "$result" ]]; then
    echo "Skipping existing result: $result"
    return
  fi
  echo "=== $label ==="
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" -m interactiondynamics.train quick "$@" \
    --dataset synthetic \
    --synthetic-num-nodes "$NUM_NODES" \
    --epochs "$EPOCHS" \
    --num-bins "$NUM_BINS" \
    --rollout-horizon "$HORIZON" \
    --synthetic-drive-cutoff "$CUTOFF" \
    --synthetic-free-rollout \
    --synthetic-self-free-rollout \
    --ift-variants auto \
    --ift-orders 2 \
    --save-jsonl "$result" \
    2>&1 | tee "$log"
}

# 1 + 4. Fair input-matched driven/free/self-free panel on every topology.
# Training is ordinary one-step driven training.  All model families in the
# quick panel are evaluated under all three input protocols.
for dynamic in "${DYNAMICS[@]}"; do
  for topology in "${TOPOLOGIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_one "second_order_${dynamic}_${topology}_seed${seed}_driven_free" \
        --synthetic-task "$dynamic" --synthetic-topology "$topology" --seed "$seed" \
        --rollout-train-steps 1
    done
  done
done

# 2. Mixed self-free-response training.  Only the base FNN is needed here:
# max-runs=1 selects ift2_auto before history variants and ordinary baselines.
for dynamic in "${DYNAMICS[@]}"; do
  for topology in "${TOPOLOGIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for percent in "${FREE_TRAIN_PERCENTS[@]}"; do
        run_one "second_order_${dynamic}_${topology}_seed${seed}_trainroll5_freetrain${percent}" \
          --synthetic-task "$dynamic" --synthetic-topology "$topology" --seed "$seed" \
          --rollout-train-steps 5 --synthetic-free-train-percent "$percent" \
          --synthetic-free-train-cutoff 1 --max-runs 1
      done
    done
  done
done

# 3. Differentiable rollout-training-length ablation for the base FNN.
for dynamic in "${DYNAMICS[@]}"; do
  for topology in "${TOPOLOGIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for unroll in "${UNROLLS[@]}"; do
        run_one "second_order_${dynamic}_${topology}_seed${seed}_trainroll${unroll}" \
          --synthetic-task "$dynamic" --synthetic-topology "$topology" --seed "$seed" \
          --rollout-train-steps "$unroll" --max-runs 1
      done
    done
  done
done

echo "Done. Stage 5 diagnostics are available per run in rollout_test, rollout_free_test, and rollout_self_free_test, each with rollout_by_step metrics."
