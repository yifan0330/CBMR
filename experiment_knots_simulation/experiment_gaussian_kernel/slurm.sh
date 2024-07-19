#!/bin/bash
#
#SBATCH --mail-user=yifan.yu@keble.ox.ac.uk
#SBATCH --mail-type=NONE
#
#SBATCH --job-name=CBMR
#SBATCH --output=/well/nichols/users/pra123/CBMR_biostats_paper/experiment_gaussian_kernel/logs/out/slurm-%A_%a.out
#SBATCH --error=/well/nichols/users/pra123/CBMR_biostats_paper/experiment_gaussian_kernel/logs/e/slurm-%A_%a.e
# Note these paths NEED TO EXIST (i.e. create logs directory before running script)
#
#SBATCH --partition=gpu_long
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --cpus-per-gpu=4


if [ -z "$SLURM_ARRAY_TASK_ID" ]
then
    # Not in Slurm Job Array - running in single mode

    JOB_ID=$SLURM_JOB_ID

    # Just read in what was passed over cmdline
    JOB_CMD="${@}"
else
    # In array

    JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

    # Get the line corresponding to the task id
    JOB_CMD=$(head -n ${SLURM_ARRAY_TASK_ID} "$1" | tail -1)
fi

# Activate the environment
source ~/.bashrc 
cd CBMR_multiple_group/20_datasets_experiment
source /well/nichols/users/pra123/anaconda3/bin/activate torch

# Train the model
srun python $JOB_CMD

exit