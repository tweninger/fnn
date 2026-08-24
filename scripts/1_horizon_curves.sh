#!/usr/bin/env bash
# Stage 1: canonical physical-event dynamics × topology benchmark.
# 3 dynamics × 5 topologies × 5 seeds × 7 models = 525 model runs.
set -euo pipefail

PYTHON="${PYTHON:-venv/bin/python}"
GPU_IDS="${GPU_IDS:-0 1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
RESULTS_DIR="${RESULTS_DIR:-derived/results/horizon_curves}"
LOG_DIR="${LOG_DIR:-derived/logs/horizon_curves}"

DYNAMICS=(diffusion wave coupled_oscillator)
TOPOLOGIES=(ring grid torus doorway swisscheese)
SEEDS=(0 1 2 3 4)
MODELS=7
EPOCHS="${EPOCHS:-20}"
NUM_NODES="${NUM_NODES:-64}"
NUM_EPISODES="${NUM_EPISODES:-10}"
NUM_BINS="${NUM_BINS:-72}"
EVENTS_PER_BIN="${EVENTS_PER_BIN:-257}"
RAINDROP_INTERVAL="${RAINDROP_INTERVAL:-12}"
EVENT_THRESHOLD="${EVENT_THRESHOLD:-0.03}"
ROLLOUT_TRAIN_STEPS="${ROLLOUT_TRAIN_STEPS:-1}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-20}"
SYNTHETIC_DT="${SYNTHETIC_DT:-0.10}"
SYNTHETIC_GAMMA="${SYNTHETIC_GAMMA:-0.15}"
SYNTHETIC_OMEGA="${SYNTHETIC_OMEGA:-0.80}"
SYNTHETIC_FORCE_SCALE="${SYNTHETIC_FORCE_SCALE:-0.80}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
read -r -a GPU_ID_LIST <<< "$GPU_IDS"

tag() { echo "${1/./p}"; }
TAU_TAG="$(tag "$EVENT_THRESHOLD")"; DT_TAG="$(tag "$SYNTHETIC_DT")"
GAMMA_TAG="$(tag "$SYNTHETIC_GAMMA")"; OMEGA_TAG="$(tag "$SYNTHETIC_OMEGA")"; FORCE_TAG="$(tag "$SYNTHETIC_FORCE_SCALE")"

active=0; launched=0
wait_for_slot() { if (( active >= MAX_PARALLEL )); then wait -n; active=$((active - 1)); fi; }

run_one() {
  local gpu="$1" dynamic="$2" topology="$3" seed="$4"
  local label="horizons_${dynamic}_${topology}_models${MODELS}_epochs${EPOCHS}_nodes${NUM_NODES}_episodes${NUM_EPISODES}_bins${NUM_BINS}_events${EVENTS_PER_BIN}_dropint${RAINDROP_INTERVAL}_tau${TAU_TAG}_trainroll${ROLLOUT_TRAIN_STEPS}_rollhorizon${ROLLOUT_HORIZON}_dt${DT_TAG}_gamma${GAMMA_TAG}_omega${OMEGA_TAG}_forcescale${FORCE_TAG}_seed${seed}"
  local result="$RESULTS_DIR/${label}.jsonl" log="$LOG_DIR/${label}.log"
  [[ -e "$result" ]] && { echo "Skipping existing: $result"; return; }
  echo "=== $dynamic / $topology / seed=$seed / gpu=$gpu ==="
  local physics_args=(--synthetic-dt "$SYNTHETIC_DT" --synthetic-gamma "$SYNTHETIC_GAMMA" --synthetic-force-scale "$SYNTHETIC_FORCE_SCALE")
  # Diffusion is first order: omega is not a parameter of its generator.
  [[ "$dynamic" != diffusion ]] && physics_args+=(--synthetic-omega "$SYNTHETIC_OMEGA")
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m interactiondynamics.train quick \
    --dataset synthetic --synthetic-task "$dynamic" --synthetic-topology "$topology" \
    --synthetic-num-nodes "$NUM_NODES" --synthetic-num-episodes "$NUM_EPISODES" \
    --synthetic-events-per-bin "$EVENTS_PER_BIN" --synthetic-raindrop-interval "$RAINDROP_INTERVAL" \
    --synthetic-event-threshold "$EVENT_THRESHOLD" "${physics_args[@]}" \
    --seed "$seed" --epochs "$EPOCHS" --max-runs "$MODELS" --num-bins "$NUM_BINS" \
    --rollout-train-steps "$ROLLOUT_TRAIN_STEPS" --rollout-horizon "$ROLLOUT_HORIZON" --debug-timing \
    --save-jsonl "$result" 2>&1 | tee "$log"
}

for dynamic in "${DYNAMICS[@]}"; do for topology in "${TOPOLOGIES[@]}"; do for seed in "${SEEDS[@]}"; do
  wait_for_slot
  gpu="${GPU_ID_LIST[$((launched % ${#GPU_ID_LIST[@]}))]}"
  run_one "$gpu" "$dynamic" "$topology" "$seed" &
  active=$((active + 1)); launched=$((launched + 1))
done; done; done
while (( active > 0 )); do wait -n; active=$((active - 1)); done
echo "Done: ${launched} datasets × ${MODELS} models = $((launched * MODELS)) model runs."
