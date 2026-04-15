# Learning-based Collision Detection in Cluttered Environments

This project focuses on implementing learning-based collision detection for robotic manipulators in cluttered environments, specifically comparing it with traditional geometric collision checkers.

## Repository Structure

- **`assets/`**: Project documentation and proposals (e.g., project proposal PDF).
- **`checkpoints/`**: Saved model weights for learning-based approaches.
- **`configs/`**: YAML configuration files for different simulation settings.
- **`data/`**: Datasets for training and evaluation.
- **`results/`**: Output from evaluations or benchmarks.
- **`scripts/`**: Utility scripts for data generation, training, and evaluation.
- **`src/`**: Core source code:
    - **`env/`**: PyBullet environment setup and obstacle spawning logic.
    - **`planner/`**: RRT-Connect planner and geometric collision detection.
    - **`model/`**: Learning-based model architectures (e.g., neural networks).
    - **`utils/`**: Helper functions for logging, visualization, and table formatting.
    - **`main.py`**: Entry point for running simulations, benchmarking, and visualization.
- **`tests/`**: Unit and integration tests for environment and collision logic.

## Setup

### Environment
The project uses Conda for dependency management. Create and activate the environment using:

```bash
conda env create -f environment.yml
conda activate cs558-project
```

Alternatively, install the required packages using pip:

```bash
pip install -e .
```

Dependencies include `pybullet`, `torch`, `numpy`, `open3d`, and `trimesh`.

## Running Simulations

The `src/main.py` script is the primary entry point for running RRT-Connect planning simulations and benchmarks.

### Basic Usage

Run a single simulation with the GUI and sparse clutter:

```bash
python -m src.main --density sparse
```

### Benchmarking

To run a full benchmark across all clutter densities (sparse, interspersed, dense) and generate plots:

```bash
python -m src.main --density all --num-paths 5 --log --plot
```

### Headless Mode

For running benchmarks without the PyBullet GUI:

```bash
python -m src.main --headless --density all --num-paths 10 --log --plot
```

### Options/Flags

- `--density`: Choose from `sparse`, `interspersed`, `dense`, or `all` for benchmarking.
- `--num-paths`: Number of planning runs per density (default: 3).
- `--headless`: Run simulation without the GUI.
- `--log`: Save planning metrics (time, success rate, etc.) to JSON in `src/logs/`.
- `--plot`: Generate result figures in `src/figures/`.
- `--seed`: Set a random seed for reproducibility.

## Testing

To verify the environment and obstacle spawning, run the interactive test scripts in the `tests/` directory:

```bash
python tests/test_env.py
python tests/test_clutter.py
```

*Note: These scripts will open a PyBullet GUI window.*

`Tristan Brideweser & Dylan Li`