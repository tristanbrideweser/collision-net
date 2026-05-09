#!/bin/bash
#SBATCH --job-name=collisionnet_bench
#SBATCH --partition=scholar-gpu
#SBATCH --account=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-bench-%j.out
#SBATCH --error=slurm-bench-%j.err

# 1. Environment Setup
module purge
module load anaconda/2024.02-py311
source venv/bin/activate

# 2. Set PYTHONPATH so imports work correctly
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 3. Paths (run from repo root via `sbatch scripts/benchmark.slurm`)
PROJ_ROOT="$(pwd)"
PKL="$PROJ_ROOT/data/envs/hybrid_solvable_problems.pkl"
MODEL="$PROJ_ROOT/src/model/checkpoints/generalist.pth"
OUT="$PROJ_ROOT/scripts/benchmark_results_${SLURM_JOB_ID}.json"

echo "🚀 Starting CollisionNet benchmark"
echo "   pkl:   $PKL"
echo "   model: $MODEL"
echo "   out:   $OUT"
nvidia-smi || true

srun python "$PROJ_ROOT/scripts/run_benchmark.py" \
    --pkl "$PKL" \
    --model "$MODEL" \
    --n-scenes 50 \
    --problems-per-scene 5 \
    --threshold 0.5 \
    --seed 0 \
    --out "$OUT"

echo "✓ Benchmark finished. Results: $OUT"