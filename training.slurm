#!/bin/bash
#SBATCH --job-name=collisionnet_sprint
#SBATCH --partition=scholar-gpu
#SBATCH --account=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# 1. Environment Setup
module purge
module load anaconda/2024.02-py311
source venv/bin/activate

# 2. Set PYTHONPATH so imports work correctly
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 3. Define the ABSOLUTE path to your data
ABS_DATA_DIR="/scratch/scholar/tbridewe/cs55800-project/data/scenes"

echo "🚀 Starting Single-GPU Training with data at: $ABS_DATA_DIR"

# 4. Launch with torchrun (matched to 1 GPU)
srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    src/train.py \
    --data-dir "$ABS_DATA_DIR" \
    --batch-size 256 \
    --num-workers 4