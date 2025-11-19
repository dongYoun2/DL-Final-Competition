#!/bin/bash

#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=test_sbatch
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dl5635@nyu.edu
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error=logs/%x/%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90            # Send signal 90s before timeout


cd /home/dl5635/projects/DL-Final-Competition
mkdir -p logs/${SLURM_JOB_NAME}

rsync -av greene-dtn:/home/$USER/projects/DL-Final-Competition /home/$USER/projects/

OVERLAY=/scratch/$USER/overlay-25GB-500K.ext3
SINGULARITY_IMAGE=/scratch/$USER/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif

singularity exec --nv --bind /scratch/$USER/ \
        --overlay $OVERLAY:ro \
        $SINGULARITY_IMAGE \
        /bin/bash -c "
        source ~/.bashrc
        conda activate ssl-vision
        python test_sbatch.py
    "
