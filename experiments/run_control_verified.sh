#!/bin/bash
#SBATCH --job-name=control_verified
#SBATCH --output=logs/control_verified_%j.log
#SBATCH --error=logs/control_verified_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Load necessary modules
module load anaconda
module load cuda/12.3.1

# Activate Virtual Environment
source venv/bin/activate

# Set Resolution
export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu

# Add pip-installed NVIDIA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH
export JAX_ENABLE_X64=True

echo "Starting Verified Control Run at $(date)"
echo "Parameters: r_drag=0.05, kappa_gm=2000.0"
echo "Job ID: $SLURM_JOB_ID"

# Run the simulation for 10 years to ensure stability
stdbuf -o0 -e0 python experiments/control_run.py --years 10.0 --r_drag 0.05 --kappa_gm 2000.0 --suffix verified

echo "Run finished at $(date)"
