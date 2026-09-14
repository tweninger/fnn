#!/usr/bin/env bash
# Experiment 6: full real-data benchmark panels.
#
# Each dataset/model pair is a separate process with its own results and log.
# CPU example: GPU_IDS=-1 CPU_THREADS=4 MAX_PARALLEL=3 BENCHMARKS=college_msg ...
set -euo pipefail
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
  echo "This parallel launcher requires Bash 5.1 or newer." >&2
  exit 2
fi

PYTHON="${PYTHON:-venv/bin/python}"
GPU_IDS="${GPU_IDS:-0 1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
CPU_THREADS="${CPU_THREADS:-1}"
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
[[ "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo "CPU_THREADS must be a positive integer" >&2; exit 2; }
(( MAX_RUNS >= 1 && MAX_RUNS <= 12 )) || {
  echo "MAX_RUNS must be between 1 and 12" >&2; exit 2;
}
(( RUN_OFFSET >= 0 && RUN_OFFSET + MAX_RUNS <= 12 )) || {
  echo "RUN_OFFSET + MAX_RUNS must select between 1 and 12 panel runs" >&2; exit 2;
}

active_jobs=0
launch_count=0
failed=0
declare -A job_pids=()
echo "Workers: ${MAX_PARALLEL} concurrent model jobs, ${CPU_THREADS} CPU threads/job (up to $((MAX_PARALLEL * CPU_THREADS)) CPU threads)."
wait_for_one() {
  local pid finished status=0
  # Reap already-finished workers explicitly (including skipped result files).
  # wait -n alone can miss jobs that finished before it was called.
  for pid in "${!job_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || status=$?
      finished="$pid"
      break
    fi
  done
  if [[ -z "${finished:-}" ]]; then
    wait -n -p finished "${!job_pids[@]}" || status=$?
  fi
  if (( status != 0 )); then
    echo "FAILED: ${job_pids[$finished]} (exit ${status}); see its log." >&2
    failed=1
  else
    echo "Finished: ${job_pids[$finished]}"
  fi
  unset 'job_pids[$finished]'
  active_jobs=$((active_jobs - 1))
}
wait_for_slot() {
  if (( active_jobs >= MAX_PARALLEL )); then
    wait_for_one
  fi
}

run_one() {
  local gpu_id="$1" benchmark="$2" model_offset="$3"
  local label="real_${benchmark}_seed${SEED}_top${TOPOLOGY_EPOCHS}_phys${PHYSICAL_EPOCHS}_cycles${CYCLES}_epochs${EPOCHS}_runs1_from${model_offset}"
  local result="$RESULTS_DIR/${label}.jsonl"
  local log="$LOG_DIR/${label}.log"
  [[ -e "$result" ]] && { echo "Skipping existing: $result"; return; }

  echo "=== real-data ${benchmark} | model offset=${model_offset}, seed=${SEED}, gpu=${gpu_id}, cpu_threads=${CPU_THREADS} | log=${log} ==="
  OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" OPENBLAS_NUM_THREADS="$CPU_THREADS" CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" -u -m interactiondynamics.train quick \
    --dataset "$benchmark" --seed "$SEED" \
    --epochs "$EPOCHS" --run-offset "$model_offset" --max-runs 1 --eval-every "$EVAL_EVERY" \
    --rollout-horizon "$ROLLOUT_HORIZON" \
    --fnn-alternating-recovery \
    --fnn-alternating-topology-epochs "$TOPOLOGY_EPOCHS" \
    --fnn-alternating-physical-epochs "$PHYSICAL_EPOCHS" \
    --fnn-alternating-cycles "$CYCLES" \
    --save-jsonl "$result" > "$log" 2>&1
}

for benchmark in "${BENCHMARKS[@]}"; do
  for ((model_offset=RUN_OFFSET; model_offset<RUN_OFFSET+MAX_RUNS; model_offset++)); do
    wait_for_slot
    gpu_id="${GPU_ID_LIST[$((launch_count % ${#GPU_ID_LIST[@]}))]}"
    run_one "$gpu_id" "$benchmark" "$model_offset" &
    job_pids[$!]="${benchmark} model offset ${model_offset}"
    active_jobs=$((active_jobs + 1))
    launch_count=$((launch_count + 1))
  done
done

while (( active_jobs > 0 )); do
  wait_for_one
done
echo "Done: ${launch_count} real-data benchmark jobs."
exit "$failed"
