#!/bin/bash
#SBATCH --job-name=collisionnet_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4                # Request 4 GPUs (Adjust if needed)
#SBATCH --cpus-per-task=32          # Request 32 CPU cores
#SBATCH --mem=128G                  # Request 128GB RAM
#SBATCH --time=04:00:00             # Max runtime set to 4 hours
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=gpu             # Target the GPU partition

# Activate environment
module purge
module load anaconda/2024.02-py311
source venv/bin/activate

# Determine GPUs dynamically
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
echo "Igniting torchrun on $NUM_GPUS GPUs..."

cd src

# Launch Distributed Training
srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train.py --batch-size 128 --epochs 50