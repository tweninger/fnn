#!/usr/bin/env bash
# Mixed free-response training for the existing second-order study.
#
# Existing driven, cutoff/self-free, and rollout-training-length runs are reused
# as controls. This launcher runs only the genuinely new 25% and 50% mixed
# self-free training conditions; every result still records driven, free/oracle,
# and self-free evaluation blocks for the new training regime.
set -euo pipefail

PYTHON="${PYTHON:-venv/bin/python}"
# Space-separated physical GPU IDs. GPU_ID remains a compatible one-device
# fallback for existing invocations.
GPU_IDS="${GPU_IDS:-${GPU_ID:-0}}"
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
FREE_TRAIN_PERCENTS=(25 50 75)

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

read -r -a GPU_ID_LIST <<< "$GPU_IDS"
if (( ${#GPU_ID_LIST[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU ID" >&2
  exit 2
fi

# Default to one independent experiment per GPU. Increase this only after
# confirming that a single job does not saturate either GPU.
MAX_PARALLEL="${MAX_PARALLEL:-${#GPU_ID_LIST[@]}}"
if (( MAX_PARALLEL < 1 )); then
  echo "MAX_PARALLEL must be at least 1; got $MAX_PARALLEL" >&2
  exit 2
fi

active_jobs=0
launch_count=0

wait_for_slot() {
  if (( active_jobs >= MAX_PARALLEL )); then
    wait -n
    active_jobs=$((active_jobs - 1))
  fi
}

run_one() {
  local gpu_id="$1"
  local label="$2"
  shift 2
  local result="$RESULTS_DIR/${label}.jsonl"
  local log="$LOG_DIR/${label}.log"
  if [[ -e "$result" ]]; then
    echo "Skipping existing result: $result"
    return
  fi
  echo "=== $label ==="
  CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" -m interactiondynamics.train quick "$@" \
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

# Mixed self-free-response training. max-runs=4 selects the four FNN rows
# (auto plus history-velocity k=1/2/3), excluding ordinary baselines.
for seed in "${SEEDS[@]}"; do
  for dynamic in "${DYNAMICS[@]}"; do
    for topology in "${TOPOLOGIES[@]}"; do
      for percent in "${FREE_TRAIN_PERCENTS[@]}"; do
        wait_for_slot
        gpu_id="${GPU_ID_LIST[$((launch_count % ${#GPU_ID_LIST[@]}))]}"
        run_one "$gpu_id" "second_order_${dynamic}_${topology}_seed${seed}_trainroll5_freetrain${percent}" \
          --synthetic-task "$dynamic" --synthetic-topology "$topology" --seed "$seed" \
          --rollout-train-steps 5 --synthetic-free-train-percent "$percent" \
          --synthetic-free-train-cutoff 1 --max-runs 6 &
        active_jobs=$((active_jobs + 1))
        launch_count=$((launch_count + 1))
      done
    done
  done
done

while (( active_jobs > 0 )); do
  wait -n
  active_jobs=$((active_jobs - 1))
done

echo "Done. Each new result includes rollout_test, rollout_free_test, and rollout_self_free_test with per-step metrics."
