#!/usr/bin/env bash
# Experiment 5: oracle-topology alternating physical recovery.
#
# The synthetic topology is fixed at truth. One FNN then learns each
# identifiable scalar one at a time; the scalar sequence repeats for the
# requested number of cycles. Diffusion has no omega phase because its
# first-order update does not depend on omega.
set -euo pipefail

PYTHON="${PYTHON:-venv/bin/python}"
GPU_IDS="${GPU_IDS:-0 1}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
RESULTS_DIR="${RESULTS_DIR:-derived/results/alternating_parameter_recovery}"
LOG_DIR="${LOG_DIR:-derived/logs/alternating_parameter_recovery}"
read -r -a SEEDS <<< "${SEEDS:-0}"
INIT_CASES="${INIT_CASES:-5}"
INIT_MIN_MULTIPLIER="${INIT_MIN_MULTIPLIER:-0.5}"
INIT_MAX_MULTIPLIER="${INIT_MAX_MULTIPLIER:-2.0}"

PHYSICAL_EPOCHS="${PHYSICAL_EPOCHS:-1}"  # per scalar subphase
CYCLES="${CYCLES:-100}"
NUM_NODES="${NUM_NODES:-128}"
NUM_EPISODES="${NUM_EPISODES:-10}"
NUM_BINS="${NUM_BINS:-72}"
EVENTS_PER_BIN="${EVENTS_PER_BIN:-513}"
RAINDROP_INTERVAL="${RAINDROP_INTERVAL:-12}"
RECOVERY_LR="${RECOVERY_LR:-0.03}"
ROLLOUT_HORIZON=1

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
read -r -a GPU_ID_LIST <<< "$GPU_IDS"
(( ${#GPU_ID_LIST[@]} > 0 )) || { echo "GPU_IDS must contain at least one GPU ID" >&2; exit 2; }
(( MAX_PARALLEL > 0 )) || { echo "MAX_PARALLEL must be at least one" >&2; exit 2; }

tag() { echo "${1/./p}"; }
sample_initialization() {
  local dynamic="$1" seed="$2" case_index="$3"
  "$PYTHON" -c '
import math
import random
import sys
dynamic, seed, case_index, lower, upper = sys.argv[1:]
truth = (0.10, 0.18, 0.70, 0.80) if dynamic == "diffusion" else (0.10, 0.15, 0.80, 0.80)
rng = random.Random(10_000 * int(seed) + 100 * int(case_index) + (0 if dynamic == "diffusion" else 1))
def sample(value):
    return value * math.exp(rng.uniform(math.log(float(lower)), math.log(float(upper))))
dt = truth[0]
_, gamma, omega, force = (sample(value) for value in truth)
if dynamic == "diffusion":
    omega = truth[2]
print(f"{dt:.8g}:{gamma:.8g}:{omega:.8g}:{force:.8g}")
' "$dynamic" "$seed" "$case_index" "$INIT_MIN_MULTIPLIER" "$INIT_MAX_MULTIPLIER"
}
active_jobs=0
launch_count=0
wait_for_slot() { if (( active_jobs >= MAX_PARALLEL )); then wait -n; active_jobs=$((active_jobs - 1)); fi; }

run_one() {
  local gpu_id="$1" dynamic="$2" case_name="$3" dt_init="$4" gamma_init="$5" omega_init="$6" force_init="$7" seed="$8"
  local true_dt true_gamma true_omega true_force scalar_count epochs physics_args label result log
  case "$dynamic" in
    diffusion) true_dt=0.10; true_gamma=0.18; true_omega=0.70; true_force=0.80; scalar_count=2 ;;
    wave) true_dt=0.10; true_gamma=0.15; true_omega=0.80; true_force=0.80; scalar_count=3 ;;
    *) echo "Unsupported dynamic: $dynamic" >&2; return 2 ;;
  esac
  epochs=$(( CYCLES * scalar_count * PHYSICAL_EPOCHS ))
  physics_args=(--synthetic-dt 0.10 --synthetic-gamma "$true_gamma" --synthetic-force-scale "$true_force")
  [[ "$dynamic" != diffusion ]] && physics_args+=(--synthetic-omega "$true_omega")
  label="alternating_${dynamic}_ring_${case_name}_dtinit$(tag "$dt_init")_gammainit$(tag "$gamma_init")_omegainit$(tag "$omega_init")_forceinit$(tag "$force_init")_lr$(tag "$RECOVERY_LR")_top0_phys${PHYSICAL_EPOCHS}_cycles${CYCLES}_epochs${epochs}_nodes${NUM_NODES}_episodes${NUM_EPISODES}_bins${NUM_BINS}_events${EVENTS_PER_BIN}_dropint${RAINDROP_INTERVAL}_seed${seed}"
  result="$RESULTS_DIR/${label}.jsonl"; log="$LOG_DIR/${label}.log"
  [[ -e "$result" ]] && { echo "Skipping existing: $result"; return; }

  echo "=== alternating ${dynamic}/${case_name} | seed=${seed}, gpu=${gpu_id} ==="
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" -u -m interactiondynamics.train quick \
    --dataset synthetic --synthetic-task "$dynamic" --synthetic-topology ring \
    --synthetic-num-nodes "$NUM_NODES" --synthetic-num-episodes "$NUM_EPISODES" \
    --synthetic-events-per-bin "$EVENTS_PER_BIN" --synthetic-raindrop-interval "$RAINDROP_INTERVAL" \
    --synthetic-event-threshold 0 "${physics_args[@]}" --seed "$seed" --epochs "$epochs" \
    --max-runs 1 --num-bins "$NUM_BINS" --rollout-train-steps 1 --rollout-horizon "$ROLLOUT_HORIZON" \
    --fnn-force-decoder field_difference --fnn-learn-physical-params --fnn-oracle-topology --fnn-alternating-recovery \
    --fnn-physical-recovery-lr "$RECOVERY_LR" \
    --fnn-alternating-physical-epochs "$PHYSICAL_EPOCHS" \
    --fnn-alternating-cycles "$CYCLES" \
    --fnn-dt "$dt_init" --fnn-gamma-init "$gamma_init" --fnn-omega-init "$omega_init" --fnn-force-scale-init "$force_init" \
    --save-jsonl "$result" 2>&1 | tee "$log"
}

for seed in "${SEEDS[@]}"; do
  for dynamic in diffusion wave; do
    for ((case_index = 0; case_index < INIT_CASES; case_index++)); do
      spec="$(sample_initialization "$dynamic" "$seed" "$case_index")"
      IFS=: read -r dt_init gamma_init omega_init force_init <<< "$spec"
      case_name="random${case_index}"
      wait_for_slot
      gpu_id="${GPU_ID_LIST[$((launch_count % ${#GPU_ID_LIST[@]}))]}"
      run_one "$gpu_id" "$dynamic" "$case_name" "$dt_init" "$gamma_init" "$omega_init" "$force_init" "$seed" &
      active_jobs=$((active_jobs + 1))
      launch_count=$((launch_count + 1))
    done
  done
done
while (( active_jobs > 0 )); do wait -n; active_jobs=$((active_jobs - 1)); done
echo "Done: ${launch_count} within-run alternating-recovery runs."
