#!/bin/bash
#SBATCH --job-name=chronos_vis_all
#SBATCH --output=logs/vis_all_%j.log
#SBATCH --error=logs/vis_all_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

module load anaconda
module load cuda/12.3.1
source venv/bin/activate
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

python experiments/plot_all_variables.py
