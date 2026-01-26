#!/bin/bash
#SBATCH --job-name=verify_jax
#SBATCH --output=logs/verify_%j.log
#SBATCH --error=logs/verify_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00

module load anaconda
module load cuda/12.3.1
source venv/bin/activate

export JAX_PLATFORM_NAME=gpu

# Add pip-installed NVIDIA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

# Minimal check
python -c "import os; print('Start Python'); import jax; print('JAX Imported'); print(jax.devices()); print('Done')"
