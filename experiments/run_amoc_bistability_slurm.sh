#!/bin/bash
#SBATCH --job-name=amoc_bist
#SBATCH --output=logs/amoc_bist_%j.log
#SBATCH --error=logs/amoc_bist_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpupreempt
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00   # gpupreempt MaxWall 30 d; run_amoc_bistability.py auto-resumes.

# AMOC bistability test (fixed-hosing, on- vs off-IC). Forwards all args.
#   sbatch --requeue experiments/run_amoc_bistability_slurm.sh \
#       --ckpt <state_base> --hosing-sv 0.57 --haline-gain 6 --k-vel 6e-5 \
#       --contrast-depth 300 --years 100 --outdir /p/tmp/$USER/amoc_bist/on

module load python/3.12.3
module load cuda/12.6.0
source venv/bin/activate
export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

mkdir -p logs
echo "Starting AMOC bistability test at $(date); Job ID: $SLURM_JOB_ID"
python - <<'PY'
import jax
assert any(d.platform == "gpu" for d in jax.devices()), "No GPU visible to JAX."
print(f"GPU preflight OK: {jax.devices()}")
PY
[ $? -ne 0 ] && { echo "GPU preflight FAILED."; exit 1; }

stdbuf -o0 -e0 python experiments/run_amoc_bistability.py "$@"
echo "Bistability test finished at $(date)."
