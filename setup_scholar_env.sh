#!/bin/bash
# setup_scholar_env.sh

# Exit immediately if a command exits with a non-zero status
set -e

echo "🧹 Purging existing modules to prevent conflicts..."
module purge

echo "🐍 Loading standard Python module..."
# Purdue RCAC typically uses anaconda or standard python modules. 
# We load the default python here, which is usually 3.9+ 
module load python

# Move the pip cache to scratch so it doesn't blow up your home directory quota
export PIP_CACHE_DIR=$RCAC_SCRATCH/.cache/pip
mkdir -p $PIP_CACHE_DIR

echo "📦 Creating virtual environment 'venv'..."
# If venv already exists, this won't overwrite it, but we can be safe
if [ ! -d "venv" ]; then
    python -m venv venv
else
    echo "⚠️ 'venv' directory already exists. Skipping creation."
fi

echo "🔌 Activating environment..."
source venv/bin/activate

echo "⬆️ Upgrading pip..."
pip install --upgrade pip

echo "🔨 Installing CollisionNet dependencies..."
# Core PyTorch (Pip usually pulls the correct CUDA binaries automatically)
pip install torch torchvision torchaudio

# Simulation and Geometry
pip install pybullet open3d trimesh

# Utilities and Logging
pip install tensorboard tqdm pyyaml numpy scipy scikit-learn

echo ""
echo "======================================================================"
echo "✅ Scholar cluster environment setup complete!"
echo "   To activate this environment in the future, just run:"
echo "   source venv/bin/activate"
echo "======================================================================"