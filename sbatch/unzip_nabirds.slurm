#!/bin/bash
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00-12:00:00
#SBATCH --mem=16GB
#SBATCH --job-name=unzip_nabirds
#SBATCH --output=logs/%x/%j.out
#SBATCH --error=logs/%x/%j.err


source /gpfs/home/dl5635/.bashrc
cd /gpfs/data/fieremanslab/dayne/projects/DL-Final-Competition
mkdir -p logs/${SLURM_JOB_NAME}

cd /gpfs/data/fieremanslab/dayne/data_dl

unzip -d ./NABirds nabirds.zip
