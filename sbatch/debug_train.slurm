#!/bin/bash
#SBATCH --account=csci_ga_2572-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=debug_train
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dl5635@nyu.edu
#SBATCH --output="logs/%x/%j.out"
#SBATCH --error=logs/%x/%j.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90            # Send signal 90s before timeout


cd /home/dl5635/projects/DL-Final-Competition
mkdir -p logs/${SLURM_JOB_NAME}

max_retries=20
for attempt in $(seq 1 $max_retries); do
    echo "Attempt $attempt/$max_retries: syncing directory..."
    if rsync -av greene-dtn:/home/$USER/projects/DL-Final-Competition /home/$USER/projects/; then
        echo "rsync succeeded on attempt $attempt"
        break
    else
        echo "rsync failed (attempt $attempt). Retrying in 1 second..."
        sleep 1
    fi

    if [[ $attempt -eq $max_retries ]]; then
        echo "rsync failed after $max_retries attempts. Exiting."
        exit 1
    fi
done

EXPERIMENT_ID=${SLURM_JOB_NAME}_${SLURM_JOB_ID}


echo "=================================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Experiment: ${EXPERIMENT_ID}"
echo "Start Time: $(date)"
echo "=================================================="

OVERLAY=/scratch/$USER/overlay-25GB-500K.ext3
SINGULARITY_IMAGE=/scratch/$USER/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif

singularity exec \
        --env HYDRA_FULL_ERROR=1 \
        --nv --bind /scratch/$USER/ \
        --overlay $OVERLAY:ro \
        $SINGULARITY_IMAGE \
        /bin/bash -c "
        source ~/.bashrc
        conda activate ssl-vision
        python ssl_vision/train.py \
        data=cifar10_debug \
        model=ibot_small \
        experiment_name=${EXPERIMENT_ID} \
        training.num_epochs=15 \
        training.batch_size=64 \
        checkpoint.save_frequency=2 \
        logging.log_frequency=10
    "
