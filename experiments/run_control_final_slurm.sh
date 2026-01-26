#!/bin/bash
#SBATCH --job-name=control_final
#SBATCH --output=logs/control_final_%j.log
#SBATCH --error=logs/control_final_%j.err
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
export JAX_ENABLE_X64=True

# Add pip-installed NVIDIA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

echo "Starting Final Control Run at $(date)"
echo "Parameters: r_drag=0.01, kappa_gm=2000.0 (Targeting ~15 Sv AMOC)"
echo "Job ID: $SLURM_JOB_ID"

# Run the simulation for 100 years
stdbuf -o0 -e0 python experiments/control_run.py --years 100.0 --r_drag 0.01 --kappa_gm 2000.0 --suffix final

echo "Run finished at $(date)"
