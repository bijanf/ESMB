#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

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

# Arguments: $1=years, $2=kappa_h, $3=suffix
YEARS=$1
KAPPA_H=$2
SUFFIX=$3

echo "Starting Debug Job: $SUFFIX at $(date)"
echo "Parameters: years=$YEARS, kappa_h=$KAPPA_H"

python experiments/control_run.py --years $YEARS --r_drag 0.05 --kappa_gm 1000.0 --kappa_h $KAPPA_H --suffix $SUFFIX

echo "Job finished at $(date)"
