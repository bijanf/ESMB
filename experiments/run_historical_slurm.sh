#!/bin/bash
#SBATCH --job-name=chronos_hist
#SBATCH --output=logs/historical_%j.log
#SBATCH --error=logs/historical_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Load necessary modules
module load anaconda
module load cuda/12.3.1

# Activate Virtual Environment
source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH

# Set Resolution (T31 or T63)
export CHRONOS_RESOLUTION=T31

# Set JAX to use appropriate platform (CPU or GPU)
export JAX_PLATFORM_NAME=gpu

export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH
export JAX_ENABLE_X64=True

echo "Starting Historical Reconstruction at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Run Simulation
stdbuf -o0 -e0 python experiments/historical_reconstruction.py

echo "Run finished at $(date)"
