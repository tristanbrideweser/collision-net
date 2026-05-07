#!/bin/bash
#SBATCH --job-name=collisionnet_ddp
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition=scholar-gpu
#SBATCH --account=gpu
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=64G

# 1. Environment Setup
module purge
module load anaconda/2024.02-py311
source venv/bin/activate

# 2. Set PYTHONPATH so imports work correctly
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 3. Define the ABSOLUTE path to your data
# Adjust the path below if your 'data' folder is elsewhere in scratch
ABS_DATA_DIR="/scratch/scholar/tbridewe/cs55800-project/data/scenes"

echo "🚀 Starting DDP Training with data at: $ABS_DATA_DIR"

# 4. Launch with torchrun
srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    src/train.py --data-dir "$ABS_DATA_DIR" --batch-size 128 --epochs 50