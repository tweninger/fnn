#!/usr/bin/env bash
# Experiment 5.5: non-oracle alternating topology and physical recovery.
# Synthetic counterpart to the JODIE block-coordinate schedule. Each run
# starts from a sampled topology logit and perturbed physical coefficients,
# then follows topology -> physical scalar sweep -> topology readaptation.
#
# Each run has an initial topology block, repeated physical-scalar sweeps, and
# a topology readaptation block after every sweep. Diffusion has three
# learnable scalars (dt, gamma, s); wave has four (omega, gamma, s, dt), so
# their total epoch counts are intentionally different.
set -euo pipefail

PYTHON="${PYTHON:-venv/bin/python}"
GPU_IDS="${GPU_IDS:-0 1}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"
RESULTS_DIR="${RESULTS_DIR:-derived/results/alternating_synthetic_recovery}"
LOG_DIR="${LOG_DIR:-derived/logs/alternating_synthetic_recovery}"

read -r -a DYNAMICS <<< "${DYNAMICS:-diffusion wave}"
read -r -a TOPOLOGIES <<< "${TOPOLOGIES:-ring grid torus doorway swisscheese}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4}"
TOPOLOGY_EPOCHS="${TOPOLOGY_EPOCHS:-50}"
PHYSICAL_EPOCHS="${PHYSICAL_EPOCHS:-10}"
CYCLES="${CYCLES:-4}"
NUM_NODES="${NUM_NODES:-64}"
NUM_EPISODES="${NUM_EPISODES:-10}"
NUM_BINS="${NUM_BINS:-72}"
EVENTS_PER_BIN="${EVENTS_PER_BIN:-257}"
RAINDROP_INTERVAL="${RAINDROP_INTERVAL:-12}"
EVENT_THRESHOLD="${EVENT_THRESHOLD:-0.03}"
ROLLOUT_HORIZON="${ROLLOUT_HORIZON:-20}"
RECOVERY_LR="${RECOVERY_LR:-0.01}"
INIT_MIN_MULTIPLIER="${INIT_MIN_MULTIPLIER:-0.5}"
INIT_MAX_MULTIPLIER="${INIT_MAX_MULTIPLIER:-2.0}"
TOPOLOGY_INIT_MIN="${TOPOLOGY_INIT_MIN:--2.0}"
TOPOLOGY_INIT_MAX="${TOPOLOGY_INIT_MAX:-2.0}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
read -r -a GPU_ID_LIST <<< "$GPU_IDS"
(( MAX_PARALLEL > 0 )) || { echo "MAX_PARALLEL must be at least one" >&2; exit 2; }

tag() { echo "${1/./p}"; }

sample_initialization() {
  local dynamic="$1" topology="$2" seed="$3"
  "$PYTHON" -c '
import math
import random
import sys
import zlib

dynamic, topology, seed, lower, upper, top_lower, top_upper = sys.argv[1:]
truth = (0.10, 0.15, 0.80, 0.80)
rng = random.Random(100_000 * int(seed) + zlib.crc32((dynamic + "/" + topology).encode()))
def sample(value):
    return value * math.exp(rng.uniform(math.log(float(lower)), math.log(float(upper))))
dt, gamma, omega, force = (sample(value) for value in truth)
if dynamic == "diffusion":
    omega = truth[2]
topology_logit = rng.uniform(float(top_lower), float(top_upper))
print(":".join(format(v, ".8g") for v in (dt, gamma, omega, force, topology_logit)))
' "$dynamic" "$topology" "$seed" "$INIT_MIN_MULTIPLIER" "$INIT_MAX_MULTIPLIER" "$TOPOLOGY_INIT_MIN" "$TOPOLOGY_INIT_MAX"
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
  local gpu_id="$1" dynamic="$2" topology="$3" seed="$4"
  local dt_init="$5" gamma_init="$6" omega_init="$7" force_init="$8" topology_init="$9"
  local scalar_count topology_epochs physical_epochs epochs physics_args
  local cycles="$CYCLES"
  case "$dynamic" in
    diffusion)
      scalar_count=3
      physics_args=(--synthetic-dt 0.10 --synthetic-gamma 0.15 --synthetic-force-scale 0.80)
      ;;
    wave)
      scalar_count=4
      physics_args=(--synthetic-dt 0.10 --synthetic-gamma 0.15 --synthetic-omega 0.80 --synthetic-force-scale 0.80)
      ;;
    *) echo "Unsupported dynamic: $dynamic" >&2; return 2 ;;
  esac
  topology_epochs="$TOPOLOGY_EPOCHS"
  physical_epochs="$PHYSICAL_EPOCHS"
  epochs=$((topology_epochs + cycles * (scalar_count * physical_epochs + topology_epochs)))

  local label result log
  label="alternatingsynth_${dynamic}_${topology}_dtinit$(tag "$dt_init")_gammainit$(tag "$gamma_init")_omegainit$(tag "$omega_init")_forceinit$(tag "$force_init")_topinit$(tag "$topology_init")_top${topology_epochs}_phys${physical_epochs}_cycles${cycles}_epochs${epochs}_nodes${NUM_NODES}_episodes${NUM_EPISODES}_bins${NUM_BINS}_events${EVENTS_PER_BIN}_dropint${RAINDROP_INTERVAL}_tau$(tag "$EVENT_THRESHOLD")_seed${seed}"
  result="$RESULTS_DIR/${label}.jsonl"
  log="$LOG_DIR/${label}.log"
  [[ -e "$result" ]] && { echo "Skipping existing: $result"; return; }
  echo "=== alternating synthetic $dynamic/$topology | seed=$seed, gpu=$gpu_id ==="
  local command=(
    env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu_id"
    "$PYTHON" -u -m interactiondynamics.train quick
    --dataset synthetic --synthetic-task "$dynamic" --synthetic-topology "$topology"
    --synthetic-num-nodes "$NUM_NODES" --synthetic-num-episodes "$NUM_EPISODES"
    --synthetic-events-per-bin "$EVENTS_PER_BIN" --synthetic-raindrop-interval "$RAINDROP_INTERVAL"
    --synthetic-event-threshold "$EVENT_THRESHOLD" "${physics_args[@]}"
    --seed "$seed" --epochs "$epochs" --max-runs 1 --num-bins "$NUM_BINS"
    --rollout-train-steps 1 --rollout-horizon "$ROLLOUT_HORIZON" --eval-every "$epochs"
    --fnn-force-decoder field_difference
    --fnn-learn-physical-params --fnn-learn-dt --fnn-alternating-recovery
    --fnn-physical-recovery-lr "$RECOVERY_LR"
    --fnn-alternating-topology-epochs "$topology_epochs"
    --fnn-alternating-physical-epochs "$physical_epochs"
    --fnn-alternating-cycles "$cycles"
    --fnn-dt "$dt_init" --fnn-gamma-init "$gamma_init"
    --fnn-omega-init "$omega_init" --fnn-force-scale-init "$force_init"
    --fnn-topology-init "$topology_init"
    --save-jsonl "$result"
  )
  "${command[@]}" 2>&1 | tee "$log"
}

for seed in "${SEEDS[@]}"; do
  for dynamic in "${DYNAMICS[@]}"; do
    for topology in "${TOPOLOGIES[@]}"; do
      spec="$(sample_initialization "$dynamic" "$topology" "$seed")"
      IFS=: read -r dt_init gamma_init omega_init force_init topology_init <<< "$spec"
      wait_for_slot
      gpu_index=$((launch_count % ${#GPU_ID_LIST[@]}))
      gpu_id="${GPU_ID_LIST[$gpu_index]}"
      run_one "$gpu_id" "$dynamic" "$topology" "$seed" \
        "$dt_init" "$gamma_init" "$omega_init" "$force_init" "$topology_init" &
      active_jobs=$((active_jobs + 1))
      launch_count=$((launch_count + 1))
    done
  done
done

while (( active_jobs > 0 )); do
  wait -n
  active_jobs=$((active_jobs - 1))
done
echo "Done: $launch_count FNN runs (2 dynamics x 5 topologies x 5 seeds)."
