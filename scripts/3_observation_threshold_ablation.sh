#!/usr/bin/env bash
# Experiment 3: observation-threshold ablation with calibrated sparsity regimes.
set -euo pipefail

PYTHON="${PYTHON:-venv/bin/python}"
GPU_IDS="${GPU_IDS:-0 1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
RESULTS_DIR="${RESULTS_DIR:-derived/results/observation_threshold_ablation}"
LOG_DIR="${LOG_DIR:-derived/logs/observation_threshold_ablation}"
CALIBRATION_CSV="$RESULTS_DIR/threshold_calibration.csv"
MANIFEST_CSV="$RESULTS_DIR/threshold_regimes.csv"

# Seed is deliberately the outermost loop so an interrupted run leaves a
# complete, immediately usable one-seed panel before proceeding to replication.
read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"
MODELS="${MODELS:-4}"
EPOCHS="${EPOCHS:-20}"
NUM_NODES="${NUM_NODES:-64}"
NUM_EPISODES="${NUM_EPISODES:-10}"
NUM_BINS="${NUM_BINS:-72}"
EVENTS_PER_BIN="${EVENTS_PER_BIN:-257}"
RAINDROP_INTERVAL="${RAINDROP_INTERVAL:-12}"
ROLLOUT_TRAIN_STEPS=1
ROLLOUT_HORIZON=20
SYNTHETIC_DT="${SYNTHETIC_DT:-0.10}"
SYNTHETIC_GAMMA="${SYNTHETIC_GAMMA:-0.15}"
SYNTHETIC_OMEGA="${SYNTHETIC_OMEGA:-0.80}"
SYNTHETIC_FORCE_SCALE="${SYNTHETIC_FORCE_SCALE:-0.80}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
read -r -a GPU_ID_LIST <<< "$GPU_IDS"
(( ${#GPU_ID_LIST[@]} > 0 )) || { echo "GPU_IDS must contain at least one GPU ID" >&2; exit 2; }
(( MAX_PARALLEL > 0 )) || { echo "MAX_PARALLEL must be at least 1" >&2; exit 2; }

"$PYTHON" scripts/event_threshold_calibration.py \
  --output-calibration "$CALIBRATION_CSV" --output-manifest "$MANIFEST_CSV" \
  --seed 0 --num-nodes "$NUM_NODES" --num-episodes "$NUM_EPISODES" \
  --num-bins "$NUM_BINS" --events-per-bin "$EVENTS_PER_BIN" \
  --raindrop-interval "$RAINDROP_INTERVAL" --synthetic-dt "$SYNTHETIC_DT" \
  --synthetic-gamma "$SYNTHETIC_GAMMA" --synthetic-omega "$SYNTHETIC_OMEGA" \
  --synthetic-force-scale "$SYNTHETIC_FORCE_SCALE"

if [[ "${CALIBRATE_ONLY:-0}" == 1 ]]; then
  echo "Calibration complete: $CALIBRATION_CSV"
  exit 0
fi

tag() { echo "${1/./p}"; }
active_jobs=0
launch_count=0
wait_for_slot() { if (( active_jobs >= MAX_PARALLEL )); then wait -n; active_jobs=$((active_jobs - 1)); fi; }

run_one() {
  local gpu_id="$1" dynamic="$2" topology="$3" regime="$4" threshold="$5" seed="$6"
  local tau_tag; tau_tag="$(tag "$threshold")"
  local label="threshold_${regime}_${dynamic}_${topology}_models${MODELS}_epochs${EPOCHS}_nodes${NUM_NODES}_episodes${NUM_EPISODES}_bins${NUM_BINS}_events${EVENTS_PER_BIN}_dropint${RAINDROP_INTERVAL}_tau${tau_tag}_trainroll${ROLLOUT_TRAIN_STEPS}_rollhorizon${ROLLOUT_HORIZON}_seed${seed}"
  local result="$RESULTS_DIR/${label}.jsonl" log="$LOG_DIR/${label}.log"
  [[ -e "$result" ]] && { echo "Skipping existing: $result"; return; }
  local physics_args=(--synthetic-dt "$SYNTHETIC_DT" --synthetic-gamma "$SYNTHETIC_GAMMA" --synthetic-force-scale "$SYNTHETIC_FORCE_SCALE")
  [[ "$dynamic" != diffusion ]] && physics_args+=(--synthetic-omega "$SYNTHETIC_OMEGA")
  echo "=== ${regime}: ${dynamic}/${topology}, tau=${threshold}, seed=${seed}, gpu=${gpu_id} ==="
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" -u -m interactiondynamics.train quick \
    --dataset synthetic --synthetic-task "$dynamic" --synthetic-topology "$topology" \
    --synthetic-num-nodes "$NUM_NODES" --synthetic-num-episodes "$NUM_EPISODES" \
    --synthetic-events-per-bin "$EVENTS_PER_BIN" --synthetic-raindrop-interval "$RAINDROP_INTERVAL" \
    --synthetic-event-threshold "$threshold" "${physics_args[@]}" --seed "$seed" \
    --epochs "$EPOCHS" --max-runs "$MODELS" --num-bins "$NUM_BINS" \
    --rollout-train-steps "$ROLLOUT_TRAIN_STEPS" --rollout-horizon "$ROLLOUT_HORIZON" \
    --save-jsonl "$result" 2>&1 | tee "$log"
}

for seed in "${SEEDS[@]}"; do
  while IFS=, read -r dynamic topology regime _target threshold _events _fraction _calibration_seed; do
    wait_for_slot
    gpu_id="${GPU_ID_LIST[$((launch_count % ${#GPU_ID_LIST[@]}))]}"
    run_one "$gpu_id" "$dynamic" "$topology" "$regime" "$threshold" "$seed" &
    active_jobs=$((active_jobs + 1))
    launch_count=$((launch_count + 1))
  done < <(tail -n +2 "$MANIFEST_CSV")
done
while (( active_jobs > 0 )); do wait -n; active_jobs=$((active_jobs - 1)); done
echo "Done: ${launch_count} datasets × ${MODELS} models = $((launch_count * MODELS)) model runs."
