#!/bin/bash
#SBATCH --job-name=collisionnet_ddp
#SBATCH --account=scholar
#SBATCH --partition=scholar-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1         # Keep as 1 for torchrun
#SBATCH --gres=gpu:2                # Request 2 GPUs per node
#SBATCH --cpus-per-task=16          # 8 cores per GPU
#SBATCH --mem=64G                   # 64GB is plenty for this data
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