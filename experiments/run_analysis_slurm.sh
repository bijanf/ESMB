#!/bin/bash
#SBATCH --job-name=chronos_analysis
#SBATCH --output=logs/analysis_%j.log
#SBATCH --error=logs/analysis_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

module load anaconda
module load cuda/12.3.1
source venv/bin/activate
export JAX_PLATFORM_NAME=cpu 
# We use CPU for analysis to save GPU hours, unless JAX forces GPU. 
# Actually, the model code imports JAX, so it might want GPU. 
# Let's keep GPU partition but we could run on CPU if needed.
# Setting platform to cpu for safety if gpu is busy? 
# No, let's use GPU for speed in calculating MOC 1200 times.
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

python experiments/plot_amoc_evolution.py
