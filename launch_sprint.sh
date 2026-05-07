#!/bin/bash

# 1. Cleanup and Directory Prep
echo "🧹 Cleaning up old state..."
scancel -u $USER
rm -rf ./checkpoints ./runs
mkdir -p ./checkpoints

# 2. Launch Job 1 (Fresh Start)
JOB1=$(sbatch --parsable training.slurm)
echo "🚀 Job 1 submitted: $JOB1"

# 3. Queue Dependencies
# afterany ensures the next job starts even if the previous one 
# timed out at the 4-hour mark (which we expect).
JOB2=$(sbatch --parsable --dependency=afterany:$JOB1 training.slurm)
echo "🔗 Job 2 queued: $JOB2"

JOB3=$(sbatch --parsable --dependency=afterany:$JOB2 training.slurm)
echo "🔗 Job 3 queued: $JOB3"

JOB4=$(sbatch --parsable --dependency=afterany:$JOB3 training.slurm)
echo "🔗 Job 4 queued: $JOB4"

JOB5=$(sbatch --parsable --dependency=afterany:$JOB4 training.slurm)
echo "🔗 Job 5 queued: $JOB5"

echo "--------------------------------------"
echo "Sprint started. Check status with: squeue --me"