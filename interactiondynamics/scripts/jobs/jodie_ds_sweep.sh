#!/bin/bash
#$ -q gpu
#$ -l gpu_card=1 
#$ -pe smp 4
#$ -N jodie_ds_sweep
#$ -t 1-2
#-o /users/akapociu/ift/interactiondynamics/logs/$JOB_NAME.$JOB_ID.$TASK_ID.out
#-e /users/akapociu/ift/interactiondynamics/logs/$JOB_NAME.$JOB_ID.$TASK_ID.err

cd /home/akapociu/ift/interactiondynamics/scripts
mkdir -p logs results

module load python
source .venv/bin/activate

export PYTHONPATH=$PWD
export PYTHONUNBUFFERED=1

# datasets=("Wikipedia" "Reddit" "MOOC" "LastFM")
datasets=("Wikipedia" "Reddit")

DATASET=${datasets[$SGE_TASK_ID-1]}

echo "Running dataset: $DATASET"

python3 scripts/hyperparam_sensitivity_sweep.py --dataset "$DATASET" --epochs 6 --results_dir results

#python3 scripts/ift_family_sweep.py --dataset "MOOC" --epochs 3 --results_dir results

