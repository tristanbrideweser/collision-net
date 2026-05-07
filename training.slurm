#!/bin/bash
#SBATCH --job-name=collisionnet_ddp
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=scholar-gpu
#SBATCH --account=gpu               # Using the account verified in your hw3 script
#SBATCH --gpus-per-node=1           # Using the specific flag that worked for you
#SBATCH --time=04:00:00

# 1. Environment Setup (RCAC Style)
module purge
module load anaconda/2024.02-py311
source venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Determine GPUs dynamically (should be 1)
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F ',' '{print NF}')
echo "Igniting torchrun on $NUM_GPUS GPU(s)..."

cd src

# 2. Launch DDP Training
# Even on 1 GPU, torchrun ensures the environment is set up for the auto-resume logic
srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    train.py --batch-size 128 --epochs 50