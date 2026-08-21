#!/usr/bin/env bash
# Experiment 4: FNN physical-parameter recovery from varied initial values.
#
# One-at-a-time starts: 13 diffusion + 18 wave + 18 coupled-oscillator
# initializations, each replicated across five seeds by default. Every JSONL
# row includes parameter_trace, so gamma/omega/force-scale recovery can be
# plotted over epochs without enabling costly per-epoch stream evaluation.
set -euo pipefail

PYTHON="${PYTHON:-venv/bin/python}"
GPU_IDS="${GPU_IDS:-0 1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
RESULTS_DIR="${RESULTS_DIR:-derived/results/parameter_recovery}"
LOG_DIR="${LOG_DIR:-derived/logs/parameter_recovery}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"
EPOCHS="${EPOCHS:-100}"
NUM_NODES="${NUM_NODES:-64}"
NUM_EPISODES="${NUM_EPISODES:-10}"
NUM_BINS="${NUM_BINS:-72}"
EVENTS_PER_BIN="${EVENTS_PER_BIN:-257}"
RAINDROP_INTERVAL="${RAINDROP_INTERVAL:-12}"
ROLLOUT_HORIZON=20

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
read -r -a GPU_ID_LIST <<< "$GPU_IDS"
(( ${#GPU_ID_LIST[@]} > 0 )) || { echo "GPU_IDS must contain at least one GPU ID" >&2; exit 2; }
(( MAX_PARALLEL > 0 )) || { echo "MAX_PARALLEL must be at least 1" >&2; exit 2; }

tag() { echo "${1/./p}"; }
active_jobs=0
launch_count=0
wait_for_slot() { if (( active_jobs >= MAX_PARALLEL )); then wait -n; active_jobs=$((active_jobs - 1)); fi; }

run_one() {
  local gpu_id="$1" dynamic="$2" parameter="$3" initial_value="$4" seed="$5"
  local true_gamma true_omega true_force init_gamma init_omega init_force init_topology
  case "$dynamic" in
    # Diffusion is first order. Omega is unused by its update but must remain
    # positive because the FNN parameterization uses a positive transform.
    diffusion) true_gamma=0.18; true_omega=0.70; true_force=0.80 ;;
    wave) true_gamma=0.15; true_omega=0.80; true_force=0.80 ;;
    coupled_oscillator) true_gamma=0.10; true_omega=1.15; true_force=0.65 ;;
    *) echo "Unknown dynamic: $dynamic" >&2; return 2 ;;
  esac
  init_gamma="$true_gamma"; init_omega="$true_omega"; init_force="$true_force"; init_topology=0.0
  case "$parameter" in
    gamma) init_gamma="$initial_value" ;;
    omega) init_omega="$initial_value" ;;
    force_scale) init_force="$initial_value" ;;
    topology) init_topology="$initial_value" ;;
    *) echo "Unknown parameter: $parameter" >&2; return 2 ;;
  esac
  local value_tag; value_tag="$(tag "$initial_value")"
  local label="paramrecovery_${dynamic}_ring_${parameter}init${value_tag}_models1_epochs${EPOCHS}_nodes${NUM_NODES}_episodes${NUM_EPISODES}_bins${NUM_BINS}_events${EVENTS_PER_BIN}_dropint${RAINDROP_INTERVAL}_tau0_trainroll1_rollhorizon${ROLLOUT_HORIZON}_seed${seed}"
  local result="$RESULTS_DIR/${label}.jsonl" log="$LOG_DIR/${label}.log"
  [[ -e "$result" ]] && { echo "Skipping existing: $result"; return; }
  local physics_args=(--synthetic-dt 0.10 --synthetic-gamma "$true_gamma" --synthetic-force-scale "$true_force")
  [[ "$dynamic" != diffusion ]] && physics_args+=(--synthetic-omega "$true_omega")
  echo "=== ${dynamic}: ${parameter}_init=${initial_value}, seed=${seed}, gpu=${gpu_id} ==="
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" -u -m interactiondynamics.train quick \
    --dataset synthetic --synthetic-task "$dynamic" --synthetic-topology ring \
    --synthetic-num-nodes "$NUM_NODES" --synthetic-num-episodes "$NUM_EPISODES" \
    --synthetic-events-per-bin "$EVENTS_PER_BIN" --synthetic-raindrop-interval "$RAINDROP_INTERVAL" \
    --synthetic-event-threshold 0 "${physics_args[@]}" --seed "$seed" --epochs "$EPOCHS" \
    --max-runs 1 --num-bins "$NUM_BINS" --rollout-train-steps 1 --rollout-horizon "$ROLLOUT_HORIZON" \
    --fnn-force-decoder field_difference --fnn-learn-physical-params \
    --fnn-gamma-init "$init_gamma" --fnn-omega-init "$init_omega" --fnn-force-scale-init "$init_force" --fnn-topology-init "$init_topology" \
    --save-jsonl "$result" 2>&1 | tee "$log"
}

for seed in "${SEEDS[@]}"; do
  for dynamic in diffusion wave; do
    case "$dynamic" in
      diffusion) starts=("gamma:0.045" "gamma:0.09" "gamma:0.18" "gamma:0.36" "gamma:0.72" "force_scale:0.20" "force_scale:0.40" "force_scale:0.80" "force_scale:1.60" "force_scale:3.20" "topology:-2" "topology:0" "topology:2") ;;
      wave) starts=("gamma:0.0375" "gamma:0.075" "gamma:0.15" "gamma:0.30" "gamma:0.60" "omega:0.20" "omega:0.40" "omega:0.80" "omega:1.60" "omega:3.20" "force_scale:0.20" "force_scale:0.40" "force_scale:0.80" "force_scale:1.60" "force_scale:3.20" "topology:-2" "topology:0" "topology:2") ;;
      coupled_oscillator) starts=("gamma:0.025" "gamma:0.05" "gamma:0.10" "gamma:0.20" "gamma:0.40" "omega:0.2875" "omega:0.575" "omega:1.15" "omega:2.30" "omega:4.60" "force_scale:0.1625" "force_scale:0.325" "force_scale:0.65" "force_scale:1.30" "force_scale:2.60" "topology:-2" "topology:0" "topology:2") ;;
    esac
    for start in "${starts[@]}"; do
      IFS=: read -r parameter initial_value <<< "$start"
      wait_for_slot
      gpu_id="${GPU_ID_LIST[$((launch_count % ${#GPU_ID_LIST[@]}))]}"
      run_one "$gpu_id" "$dynamic" "$parameter" "$initial_value" "$seed" &
      active_jobs=$((active_jobs + 1))
      launch_count=$((launch_count + 1))
    done
  done
done
while (( active_jobs > 0 )); do wait -n; active_jobs=$((active_jobs - 1)); done
echo "Done: ${launch_count} FNN recovery runs."
