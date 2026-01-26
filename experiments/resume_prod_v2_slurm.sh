#!/bin/bash
#SBATCH --job-name=prod_v2_resume
#SBATCH --output=logs/prod_v2_resume_%j.log
#SBATCH --error=logs/prod_v2_resume_%j.err
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
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=True

# Add pip-installed NVIDIA libraries
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

echo "Resuming Production Run V2 at $(date)"
echo "Parameters: r_drag=0.05, kappa_gm=1000.0, kappa_h=100.0"
echo "Restart File: outputs/control_run_prod_v2/restart_0132.nc"

# Run the simulation for 100 years (will continue from Month 133)
stdbuf -o0 -e0 python experiments/control_run.py --years 100.0 --r_drag 0.05 --kappa_gm 1000.0 --kappa_h 100.0 --suffix prod_v2 --restart_file outputs/control_run_prod_v2/restart_0132.nc

echo "Resume finished at $(date)"
