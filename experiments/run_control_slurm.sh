#!/bin/bash
#SBATCH --job-name=chronos_control
#SBATCH --output=logs/control_%j.log
#SBATCH --error=logs/control_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --qos=priority
#SBATCH --account=poem

# Load necessary modules (Adjust these based on your HPC environment)
module load cuda/11.8 # Required for GPU support
# module load python/3.10

# Activate Virtual Environment
source venv/bin/activate

# Set JAX to use appropriate platform (CPU or GPU)
export JAX_PLATFORM_NAME=gpu

# Optimization flags for JAX on CPU (if applicable)
# export XLA_FLAGS="--xla_force_host_platform_device_count=64"

echo "Starting Control Run at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# Run the simulation
# Using stdbuf to unbuffer output so we can see logs in real-time
stdbuf -o0 -e0 python experiments/control_run.py

echo "Run finished at $(date)"
