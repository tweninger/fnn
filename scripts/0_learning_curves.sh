#!/usr/bin/env bash
# Stage 0: learning-curve diagnostic for the canonical event-only wave task.
#
# Runs the minimal physical-event learning-curve comparison—FNN and the
# standard sum/TGN-GRU baseline—for five independent seeds. Each JSONL retains
# one record per epoch, including train objective, validation/test force
# metrics, active-force metrics, event AUPR, and closed-loop rollout metrics.
set -euo pipefail

PYTHON="${PYTHON:-venv/bin/python}"
# Use both physical GPUs by default. Override, for example, with GPU_IDS="2 3".
GPU_IDS="${GPU_IDS:-0 1}"
# Two experiments per GPU was the observed saturation point on wl-gpu1.
MAX_PARALLEL="${MAX_PARALLEL:-2}"

RESULTS_DIR="${RESULTS_DIR:-derived/results/learning_curves}"
LOG_DIR="${LOG_DIR:-derived/logs/learning_curves}"

# Every value below is passed explicitly and encoded in each result filename.
DYNAMIC="wave"
TOPOLOGY="ring"
# Physical quick-panel order is FNN, then sum/TGN-GRU. Keep this at two for
# convergence diagnostics; the broader seven-model comparison is Stage 1.
MODELS=2
SEEDS=(0 1 2 3 4)
EPOCHS=20
NUM_NODES=64
NUM_EPISODES=10
NUM_BINS=72
# 257 exceeds the largest directed grid/torus support at N=64 and avoids
# event-budget clipping.  It is intentionally retained here for comparability
# with the later all-topology panel.
EVENTS_PER_BIN=257
RAINDROP_INTERVAL=12
EVENT_THRESHOLD=0.03
ROLLOUT_TRAIN_STEPS=1
ROLLOUT_HORIZON=1
SYNTHETIC_DT=0.10
SYNTHETIC_GAMMA=0.15
SYNTHETIC_OMEGA=0.80
SYNTHETIC_FORCE_SCALE=0.80

# Filesystem-friendly tags that stay correct if any numeric knob above changes.
EVENT_THRESHOLD_TAG="${EVENT_THRESHOLD/./p}"
SYNTHETIC_DT_TAG="${SYNTHETIC_DT/./p}"
SYNTHETIC_GAMMA_TAG="${SYNTHETIC_GAMMA/./p}"
SYNTHETIC_OMEGA_TAG="${SYNTHETIC_OMEGA/./p}"
SYNTHETIC_FORCE_SCALE_TAG="${SYNTHETIC_FORCE_SCALE/./p}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

read -r -a GPU_ID_LIST <<< "$GPU_IDS"
if (( ${#GPU_ID_LIST[@]} == 0 )); then
  echo "GPU_IDS must contain at least one GPU ID" >&2
  exit 2
fi
if (( MAX_PARALLEL < 1 )); then
  echo "MAX_PARALLEL must be at least 1; got $MAX_PARALLEL" >&2
  exit 2
fi

LABEL_PREFIX="learningcurves_${DYNAMIC}_${TOPOLOGY}_models${MODELS}_epochs${EPOCHS}_nodes${NUM_NODES}_episodes${NUM_EPISODES}_bins${NUM_BINS}_events${EVENTS_PER_BIN}_dropint${RAINDROP_INTERVAL}_tau${EVENT_THRESHOLD_TAG}_trainroll${ROLLOUT_TRAIN_STEPS}_rollhorizon${ROLLOUT_HORIZON}_dt${SYNTHETIC_DT_TAG}_gamma${SYNTHETIC_GAMMA_TAG}_omega${SYNTHETIC_OMEGA_TAG}_forcescale${SYNTHETIC_FORCE_SCALE_TAG}"

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
  local seed="$2"
  local label="${LABEL_PREFIX}_seed${seed}"
  local result="$RESULTS_DIR/${label}.jsonl"
  local log="$LOG_DIR/${label}.log"

  if [[ -e "$result" ]]; then
    echo "Skipping existing result: $result"
    return
  fi

  echo "=== ${label} | gpu=${gpu_id} ==="
  # -u is important here: tee makes stdout a pipe, which otherwise causes
  # Python to buffer step/epoch prints until a large output block accumulates.
  CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" -u -m interactiondynamics.train quick \
    --dataset synthetic \
    --synthetic-task "$DYNAMIC" \
    --synthetic-topology "$TOPOLOGY" \
    --synthetic-num-nodes "$NUM_NODES" \
    --synthetic-num-episodes "$NUM_EPISODES" \
    --synthetic-events-per-bin "$EVENTS_PER_BIN" \
    --synthetic-raindrop-interval "$RAINDROP_INTERVAL" \
    --synthetic-event-threshold "$EVENT_THRESHOLD" \
    --synthetic-dt "$SYNTHETIC_DT" \
    --synthetic-gamma "$SYNTHETIC_GAMMA" \
    --synthetic-omega "$SYNTHETIC_OMEGA" \
    --synthetic-force-scale "$SYNTHETIC_FORCE_SCALE" \
    --seed "$seed" \
    --epochs "$EPOCHS" \
    --eval-every 1 \
    --max-runs "$MODELS" \
    --num-bins "$NUM_BINS" \
    --rollout-train-steps "$ROLLOUT_TRAIN_STEPS" \
    --rollout-horizon "$ROLLOUT_HORIZON" \
    --debug-timing \
    --save-jsonl "$result" \
    2>&1 | tee "$log"
}

for seed in "${SEEDS[@]}"; do
  wait_for_slot
  gpu_id="${GPU_ID_LIST[$((launch_count % ${#GPU_ID_LIST[@]}))]}"
  run_one "$gpu_id" "$seed" &
  active_jobs=$((active_jobs + 1))
  launch_count=$((launch_count + 1))
done

while (( active_jobs > 0 )); do
  wait -n
  active_jobs=$((active_jobs - 1))
done

echo "Done"
