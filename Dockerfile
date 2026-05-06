# Start with the official PyTorch image (matching your version + CUDA)
FROM pytorch/pytorch:2.1.1-cuda11.8-cudnn8-runtime

# Prevent interactive prompts blocking the build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required for PyBullet and 3D rendering
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up the working directory
WORKDIR /workspace/collisionnet

# Copy ONLY the dependency files first. 
# This caches the heavy pip installs unless these specific files change.
COPY requirements.txt pyproject.toml ./

# Sync your specific build tools to avoid the ROS 2 conflicts
RUN pip install --upgrade pip==21.3.1 setuptools==59.5.0 wheel==0.37.1

# Install the project dependencies
RUN pip install -r requirements.txt

# Set the default command to open a bash shell
CMD ["bash"]
