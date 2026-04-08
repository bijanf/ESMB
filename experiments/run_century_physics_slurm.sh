#!/bin/bash
#SBATCH --job-name=chronos_physics
#SBATCH --output=logs/century_v2_%j.log
#SBATCH --error=logs/century_v2_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpumedium
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=23:00:00

# Load necessary modules
module load anaconda
module load cuda/12.3.1

# Activate Virtual Environment
source venv/bin/activate

# Resolution
export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True

# NVIDIA Libs
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

echo "Starting Chronos-ESM Century Run (Physics Overhaul) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Run (fresh start or resume)
# Fresh:  stdbuf -o0 -e0 python experiments/run_century_physics.py
# Resume: stdbuf -o0 -e0 python experiments/run_century_physics.py --resume year_042
stdbuf -o0 -e0 python experiments/run_century_physics.py "$@"

echo "Run finished at $(date)"
