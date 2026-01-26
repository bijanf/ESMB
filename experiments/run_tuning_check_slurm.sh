#!/bin/bash
#SBATCH --job-name=tune_check
#SBATCH --output=logs/tune_check_%j.log
#SBATCH --error=logs/tune_check_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

module load anaconda
module load cuda/12.3.1

source venv/bin/activate
export PYTHONPATH=$(pwd):$PYTHONPATH
export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

echo "Starting Tuning Check Run at $(date)"
# Run for 2 years to see T trend
python -u experiments/control_run.py --years 2.0 --r_drag 0.05 --kappa_gm 2000.0 --suffix tuning_check
echo "Run finished at $(date)"
