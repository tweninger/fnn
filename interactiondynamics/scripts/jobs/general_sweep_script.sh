#!/bin/bash
#$ -q gpu
#$ -l gpu_card=1 
#$ -pe smp 4
#$ -N jodie_ds_sweep
#$ -t 1-2
#$ -o logs/$JOB_NAME.$JOB_ID.$TASK_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.$TASK_ID.err

cd /home/akapociu/ift/interactiondynamics/scripts
mkdir -p logs results

module load python
source .venv/bin/activate

# datasets=("Wikipedia" "Reddit" "MOOC" "LastFM")
datasets=("Wikipedia" "Reddit")

DATASET=${datasets[$SGE_TASK_ID-1]}

echo "Running dataset: $DATASET"

python ds_sweep_one_dataset.py --dataset "$DATASET" --epochs 6 --results_dir results