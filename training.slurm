#!/bin/bash
#SBATCH --job-name=collisionnet_ddp
#SBATCH --account=scholar
#SBATCH --partition=scholar-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                # Requesting exactly 1 GPU for maximum availability
#SBATCH --cpus-per-task=8           # 8 cores is plenty for one GPU
#SBATCH --mem=32G                   # 32GB is safe and respectful of the node
#SBATCH --time=04:00:00             
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# 1. Environment Setup
module purge
module load anaconda/2024.02-py311
source venv/bin/activate

# Determine GPUs dynamically
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
echo "Igniting torchrun on $NUM_GPUS GPUs..."

cd src

# 2. Launch DDP Training
srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train.py --batch-size 128 --epochs 50