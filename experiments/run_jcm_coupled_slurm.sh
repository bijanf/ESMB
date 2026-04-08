#!/bin/bash
#SBATCH --job-name=chronos_jcm
#SBATCH --output=logs/jcm_coupled_%j.log
#SBATCH --error=logs/jcm_coupled_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpumedium
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# Load necessary modules
module load anaconda
module load cuda/12.3.1

# Activate Virtual Environment
source venv/bin/activate

# Add project root to PYTHONPATH
export PYTHONPATH="${SLURM_SUBMIT_DIR:-/home/fallah/scripts/ESMB}:$PYTHONPATH"

# Resolution
export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu
# Note: x64 disabled for JCM — dinosaur/SPEEDY use float32 internally
# and float64 JIT compilation is extremely slow (30+ min)
#export JAX_ENABLE_X64=True

# NVIDIA Libs
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

echo "Starting Chronos-ESM with JCM Atmosphere at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Run (default 10 years, or pass --years N)
stdbuf -o0 -e0 python experiments/run_jcm_coupled.py "$@"

echo "Run finished at $(date)"
