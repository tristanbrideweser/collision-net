#!/bin/bash
# setup_scholar_env.sh

set -e

echo "🧹 Purging modules and loading Anaconda..."
module purge
module load anaconda/2024.02-py311

# Redirect pip cache to scratch to protect your home quota
export PIP_CACHE_DIR=$RCAC_SCRATCH/.cache/pip
mkdir -p $PIP_CACHE_DIR

echo "📦 Creating venv in Scratch space..."
# Create the virtual environment using the loaded Anaconda python
python -m venv venv
source venv/bin/activate

echo "⬆️ Upgrading pip and installing high-performance dependencies..."
pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support for the Scholar GPUs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install simulation, geometry, and logging tools
pip install pybullet numpy tqdm tensorboard scipy

echo "✅ Environment successfully mirrored to Scholar!"