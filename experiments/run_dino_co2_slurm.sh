#!/bin/bash
#SBATCH --job-name=dino_co2
#SBATCH --output=logs/dino_co2_%j.log
#SBATCH --error=logs/dino_co2_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpupreempt
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Forced CO2 experiment on the freed (q-flux) model. Forwards all args to run_dino_co2.py.
#   sbatch --requeue experiments/run_dino_co2_slurm.sh \
#       --ckpt /p/tmp/$USER/dino_control_lib/state_d036500 --co2 560 --years 50 \
#       --outdir /p/tmp/$USER/co2_2x

module load python/3.12.3
module load cuda/12.6.0
source venv/bin/activate

export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

mkdir -p logs
echo "Starting forced CO2 experiment at $(date); Job ID: $SLURM_JOB_ID"

python - <<'PY'
import jax
assert any(d.platform == "gpu" for d in jax.devices()), "No GPU visible to JAX."
print(f"GPU preflight OK: {jax.devices()}")
PY
[ $? -ne 0 ] && { echo "GPU preflight FAILED -- aborting."; exit 1; }

python experiments/prefetch_data.py --check
[ $? -ne 0 ] && { echo "Datasets not staged."; exit 1; }

stdbuf -o0 -e0 python experiments/run_dino_co2.py "$@"
echo "Run finished at $(date)."
