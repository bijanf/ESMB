#!/bin/bash
#SBATCH --job-name=chronos_resume
#SBATCH --output=logs/resume_%j.log
#SBATCH --error=logs/resume_%j.err
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

# Environment Variables
export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True

# Add pip-installed NVIDIA libraries to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

echo "Starting Resume Run at $(date)"
stdbuf -o0 -e0 python experiments/resume_run.py
echo "Run finished at $(date)"
