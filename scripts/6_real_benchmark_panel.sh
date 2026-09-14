#!/usr/bin/env bash
# Experiment 6: full real-data benchmark panels.
#
# Each requested real-data dataset runs the FNN plus every neural baseline once.
set -euo pipefail

PYTHON="${PYTHON:-venv/bin/python}"
GPU_IDS="${GPU_IDS:-0 1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
RESULTS_DIR="${RESULTS_DIR:-derived/results/real_benchmark_panel}"
LOG_DIR="${LOG_DIR:-derived/logs/real_benchmark_panel}"
read -r -a BENCHMARKS <<< "${BENCHMARKS:-college_msg email_eu_core sociopatterns}"
# The panel order is FNN, sum-GRU, DeepSets-GRU, SetTransformer-GRU, Hopfield,
# SetTransformer-LNN, SetTransformer-HNN, EdgeBank, GraphMixer, TGN, DyGFormer, JODIE.
# Set MAX_RUNS=7 to retain the original panel only.
MAX_RUNS="${MAX_RUNS:-12}"
# Set RUN_OFFSET=1 and MAX_RUNS=6 to restart after an already-completed FNN.
RUN_OFFSET="${RUN_OFFSET:-0}"
SEED="${SEED:-0}"

TOPOLOGY_EPOCHS="${TOPOLOGY_EPOCHS:-30}"
PHYSICAL_EPOCHS="${PHYSICAL_EPOCHS:-10}"
CYCLES="${CYCLES:-2}"
# One initial topology fit, followed by each physical sweep and its topology
# recovery block.  This ensures the selected run ends with the scorer adapted
# to the latest physical parameters.
EPOCHS=$(( TOPOLOGY_EPOCHS + CYCLES * (3 * PHYSICAL_EPOCHS + TOPOLOGY_EPOCHS) ))
EVAL_EVERY="${EVAL_EVERY:-10}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-20}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
read -r -a GPU_ID_LIST <<< "$GPU_IDS"
(( ${#GPU_ID_LIST[@]} > 0 )) || { echo "GPU_IDS must contain at least one GPU ID" >&2; exit 2; }
(( MAX_PARALLEL > 0 )) || { echo "MAX_PARALLEL must be at least one" >&2; exit 2; }
(( MAX_RUNS >= 1 && MAX_RUNS <= 12 )) || {
  echo "MAX_RUNS must be between 1 and 12" >&2; exit 2;
}
(( RUN_OFFSET >= 0 && RUN_OFFSET + MAX_RUNS <= 12 )) || {
  echo "RUN_OFFSET + MAX_RUNS must select between 1 and 12 panel runs" >&2; exit 2;
}

active_jobs=0
launch_count=0
wait_for_slot() {
  if (( active_jobs >= MAX_PARALLEL )); then
    wait -n
    active_jobs=$((active_jobs - 1))
  fi
}

run_one() {
  local gpu_id="$1" benchmark="$2" max_runs="$3"
  local offset_suffix=""
  (( RUN_OFFSET > 0 )) && offset_suffix="_from${RUN_OFFSET}"
  local label="real_${benchmark}_seed${SEED}_top${TOPOLOGY_EPOCHS}_phys${PHYSICAL_EPOCHS}_cycles${CYCLES}_epochs${EPOCHS}_runs${max_runs}${offset_suffix}"
  local result="$RESULTS_DIR/${label}.jsonl"
  local log="$LOG_DIR/${label}.log"
  [[ -e "$result" ]] && { echo "Skipping existing: $result"; return; }

  echo "=== real-data ${benchmark} | offset=${RUN_OFFSET}, runs=${max_runs}, seed=${SEED}, gpu=${gpu_id} ==="
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" -u -m interactiondynamics.train quick \
    --dataset "$benchmark" --seed "$SEED" \
    --epochs "$EPOCHS" --run-offset "$RUN_OFFSET" --max-runs "$max_runs" --eval-every "$EVAL_EVERY" \
    --rollout-horizon "$ROLLOUT_HORIZON" \
    --fnn-alternating-recovery \
    --fnn-alternating-topology-epochs "$TOPOLOGY_EPOCHS" \
    --fnn-alternating-physical-epochs "$PHYSICAL_EPOCHS" \
    --fnn-alternating-cycles "$CYCLES" \
    --save-jsonl "$result" 2>&1 | tee "$log"
}

for benchmark in "${BENCHMARKS[@]}"; do
  wait_for_slot
  gpu_id="${GPU_ID_LIST[$((launch_count % ${#GPU_ID_LIST[@]}))]}"
  run_one "$gpu_id" "$benchmark" "$MAX_RUNS" &
  active_jobs=$((active_jobs + 1))
  launch_count=$((launch_count + 1))
done

while (( active_jobs > 0 )); do
  wait -n
  active_jobs=$((active_jobs - 1))
done
echo "Done: ${launch_count} real-data benchmark jobs."
