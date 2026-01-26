#!/bin/bash
#SBATCH --job-name=chronos_tune
#SBATCH --output=logs/tuning_%j.log
#SBATCH --error=logs/tuning_%j.err
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

# Add pip-installed NVIDIA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH
export JAX_ENABLE_X64=True

echo "Starting Tuning Run at $(date)"
python experiments/tune_physics.py
echo "Tuning finished at $(date)"
