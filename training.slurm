#!/bin/bash
#SBATCH --job-name=collisionnet_ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4                # Scholar nodes typically have 4 GPUs
#SBATCH --cpus-per-task=32          # 8 cores per GPU is standard for Scholar
#SBATCH --mem=128G                  
#SBATCH --time=04:00:00             # Strictly 4 hours as per cluster limits
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=scholar-gpu     # <--- Updated partition name from your sinfo

# 1. Load the Anaconda environment verified earlier
module purge
module load anaconda/2024.02-py311
source venv/bin/activate

# Determine the number of GPUs Slurm allocated
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
echo "Igniting torchrun on $NUM_GPUS GPUs..."

cd src

# 2. Launch DDP Training
# The --standalone flag works for single-node multi-GPU runs
srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train.py --batch-size 128 --epochs 50