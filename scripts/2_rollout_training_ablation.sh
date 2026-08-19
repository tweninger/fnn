#!/usr/bin/env bash
# Stage 2a: rollout-training-length ablation across physical dynamics and topologies.
#
# 3 dynamics × 5 topologies × 5 training horizons × 5 seeds × 7 models =
# 2,625 model runs. Every run is evaluated at a fixed 20-step rollout horizon.
set -euo pipefail

PYTHON="${PYTHON:-venv/bin/python}"
GPU_IDS="${GPU_IDS:-0 1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
RESULTS_DIR="${RESULTS_DIR:-derived/results/rollout_training_ablation}"
LOG_DIR="${LOG_DIR:-derived/logs/rollout_training_ablation}"

DYNAMICS=(diffusion wave coupled_oscillator)
TOPOLOGIES=(ring grid torus doorway swisscheese)
TRAIN_ROLLOUT_STEPS=(1 2 5 10 20)
SEEDS=(0 1 2 3 4)
MODELS=7
EPOCHS="${EPOCHS:-20}"
NUM_NODES="${NUM_NODES:-64}"
NUM_EPISODES="${NUM_EPISODES:-10}"
NUM_BINS="${NUM_BINS:-72}"
EVENTS_PER_BIN="${EVENTS_PER_BIN:-257}"
RAINDROP_INTERVAL="${RAINDROP_INTERVAL:-12}"
EVENT_THRESHOLD="${EVENT_THRESHOLD:-0.03}"
ROLLOUT_HORIZON=20
SYNTHETIC_DT="${SYNTHETIC_DT:-0.10}"
SYNTHETIC_GAMMA="${SYNTHETIC_GAMMA:-0.15}"
SYNTHETIC_OMEGA="${SYNTHETIC_OMEGA:-0.80}"
SYNTHETIC_FORCE_SCALE="${SYNTHETIC_FORCE_SCALE:-0.80}"

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

tag() { echo "${1/./p}"; }
TAU_TAG="$(tag "$EVENT_THRESHOLD")"
DT_TAG="$(tag "$SYNTHETIC_DT")"
GAMMA_TAG="$(tag "$SYNTHETIC_GAMMA")"
OMEGA_TAG="$(tag "$SYNTHETIC_OMEGA")"
FORCE_TAG="$(tag "$SYNTHETIC_FORCE_SCALE")"

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
  local dynamic="$2"
  local topology="$3"
  local train_steps="$4"
  local seed="$5"
  local label="rollouttrain_${dynamic}_${topology}_models${MODELS}_epochs${EPOCHS}_nodes${NUM_NODES}_episodes${NUM_EPISODES}_bins${NUM_BINS}_events${EVENTS_PER_BIN}_dropint${RAINDROP_INTERVAL}_tau${TAU_TAG}_trainroll${train_steps}_rollhorizon${ROLLOUT_HORIZON}_dt${DT_TAG}_gamma${GAMMA_TAG}_omega${OMEGA_TAG}_forcescale${FORCE_TAG}_seed${seed}"
  local result="$RESULTS_DIR/${label}.jsonl"
  local log="$LOG_DIR/${label}.log"

  if [[ -e "$result" ]]; then
    echo "Skipping existing result: $result"
    return
  fi

  echo "=== ${dynamic} / ${topology} / K_train=${train_steps} / seed=${seed} / gpu=${gpu_id} ==="
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" -u -m interactiondynamics.train quick \
    --dataset synthetic \
    --synthetic-task "$dynamic" \
    --synthetic-topology "$topology" \
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
    --max-runs "$MODELS" \
    --num-bins "$NUM_BINS" \
    --rollout-train-steps "$train_steps" \
    --rollout-horizon "$ROLLOUT_HORIZON" \
    --save-jsonl "$result" \
    2>&1 | tee "$log"
}
for seed in "${SEEDS[@]}"; do
  for dynamic in "${DYNAMICS[@]}"; do
    for topology in "${TOPOLOGIES[@]}"; do
      for train_steps in "${TRAIN_ROLLOUT_STEPS[@]}"; do
        wait_for_slot
        gpu_id="${GPU_ID_LIST[$((launch_count % ${#GPU_ID_LIST[@]}))]}"
        run_one "$gpu_id" "$dynamic" "$topology" "$train_steps" "$seed" &
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

echo "Done: ${launch_count} datasets × ${MODELS} models = $((launch_count * MODELS)) model runs."
