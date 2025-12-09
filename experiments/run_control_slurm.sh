#!/bin/bash
#SBATCH --job-name=chronos_control
#SBATCH --output=logs/control_%j.log
#SBATCH --error=logs/control_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpumedium
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00

# Load necessary modules
module load anaconda
module load cuda/12.1

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
