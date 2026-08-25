#!/usr/bin/env bash
# Experiment 6: JODIE benchmark panel with one full baseline comparison.
#
# Wikipedia runs the FNN plus every neural baseline once.  The remaining JODIE
# datasets run the same alternating FNN recovery schedule alone, so they test
# transfer across datasets without multiplying the expensive baseline panel.
set -euo pipefail

PYTHON="${PYTHON:-venv/bin/python}"
GPU_IDS="${GPU_IDS:-0 1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
RESULTS_DIR="${RESULTS_DIR:-derived/results/jodie_benchmark_panel}"
LOG_DIR="${LOG_DIR:-derived/logs/jodie_benchmark_panel}"
read -r -a BENCHMARKS <<< "${BENCHMARKS:-wikipedia reddit mooc lastfm}"
FULL_BENCHMARK="${FULL_BENCHMARK:-wikipedia}"
# The panel order is FNN, sum-GRU, DeepSets-GRU, SetTransformer-GRU, Hopfield,
# SetTransformer-LNN, SetTransformer-HNN.  Seven includes every comparator;
# set this to six to omit only the final HNN run.
WIKIPEDIA_MAX_RUNS="${WIKIPEDIA_MAX_RUNS:-7}"
OTHER_MAX_RUNS="${OTHER_MAX_RUNS:-1}"
SEED="${SEED:-0}"

TOPOLOGY_EPOCHS="${TOPOLOGY_EPOCHS:-30}"
PHYSICAL_EPOCHS="${PHYSICAL_EPOCHS:-10}"
CYCLES="${CYCLES:-2}"
EPOCHS=$(( CYCLES * (TOPOLOGY_EPOCHS + 4 * PHYSICAL_EPOCHS) ))
EVAL_EVERY="${EVAL_EVERY:-10}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-20}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
read -r -a GPU_ID_LIST <<< "$GPU_IDS"
(( ${#GPU_ID_LIST[@]} > 0 )) || { echo "GPU_IDS must contain at least one GPU ID" >&2; exit 2; }
(( MAX_PARALLEL > 0 )) || { echo "MAX_PARALLEL must be at least one" >&2; exit 2; }
(( WIKIPEDIA_MAX_RUNS >= 1 && WIKIPEDIA_MAX_RUNS <= 7 )) || {
  echo "WIKIPEDIA_MAX_RUNS must be between 1 and 7" >&2; exit 2;
}
(( OTHER_MAX_RUNS == 1 )) || { echo "OTHER_MAX_RUNS must be 1 so non-Wikipedia datasets run FNN only" >&2; exit 2; }

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
  local label="jodie_${benchmark}_seed${SEED}_top${TOPOLOGY_EPOCHS}_phys${PHYSICAL_EPOCHS}_cycles${CYCLES}_epochs${EPOCHS}_runs${max_runs}"
  local result="$RESULTS_DIR/${label}.jsonl"
  local log="$LOG_DIR/${label}.log"
  [[ -e "$result" ]] && { echo "Skipping existing: $result"; return; }

  echo "=== JODIE ${benchmark} | runs=${max_runs}, seed=${SEED}, gpu=${gpu_id} ==="
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" -u -m interactiondynamics.train quick \
    --dataset jodie "$benchmark" --jodie-fnn --seed "$SEED" \
    --epochs "$EPOCHS" --max-runs "$max_runs" --eval-every "$EVAL_EVERY" \
    --rollout-horizon "$ROLLOUT_HORIZON" \
    --fnn-alternating-recovery \
    --fnn-alternating-topology-epochs "$TOPOLOGY_EPOCHS" \
    --fnn-alternating-physical-epochs "$PHYSICAL_EPOCHS" \
    --fnn-alternating-cycles "$CYCLES" \
    --save-jsonl "$result" 2>&1 | tee "$log"
}

for benchmark in "${BENCHMARKS[@]}"; do
  max_runs="$OTHER_MAX_RUNS"
  [[ "$benchmark" == "$FULL_BENCHMARK" ]] && max_runs="$WIKIPEDIA_MAX_RUNS"
  wait_for_slot
  gpu_id="${GPU_ID_LIST[$((launch_count % ${#GPU_ID_LIST[@]}))]}"
  run_one "$gpu_id" "$benchmark" "$max_runs" &
  active_jobs=$((active_jobs + 1))
  launch_count=$((launch_count + 1))
done

while (( active_jobs > 0 )); do
  wait -n
  active_jobs=$((active_jobs - 1))
done
echo "Done: ${launch_count} JODIE benchmark jobs."
