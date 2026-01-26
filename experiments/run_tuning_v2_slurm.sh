#!/bin/bash
#SBATCH --job-name=chronos_tune
#SBATCH --output=logs/tune_%j.log
#SBATCH --error=logs/tune_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00



module load anaconda
module load cuda/12.3.1

source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu
# Optimized XLA flags for memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

# Add pip-installed NVIDIA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH
export JAX_ENABLE_X64=True
export JAX_TRACEBACK_FILTERING=off





export JAX_PLATFORMS=cuda







echo "Starting Tuning V2 at $(date)"
python -u experiments/tune_physics_v2.py
echo "Tuning finished at $(date)"
