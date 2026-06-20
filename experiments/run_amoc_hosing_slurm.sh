#!/bin/bash
#SBATCH --job-name=amoc_hose
#SBATCH --output=logs/amoc_hose_%j.log
#SBATCH --error=logs/amoc_hose_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpupreempt
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00   # gpupreempt MaxWall is 30 d; a full sweep is ~4.4 h, +margin.
                         # run_amoc_hosing.py auto-resumes from per-level ckpts, so a
                         # --requeue (preemption) continues instead of re-spinning up.

# AMOC hosing / hysteresis sweep. Forwards all args to run_amoc_hosing.py.
#   sbatch --dependency=afterok:<control_jobid> --requeue experiments/run_amoc_hosing_slurm.sh \
#       --ckpt /p/tmp/$USER/dino_control_thc/state_d036500 --fmax 1.0 --nsteps 5 --hold-years 15 \
#       --outdir /p/tmp/$USER/amoc_hosing

module load python/3.12.3
module load cuda/12.6.0
source venv/bin/activate
export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

mkdir -p logs
echo "Starting AMOC hosing sweep at $(date); Job ID: $SLURM_JOB_ID"
python - <<'PY'
import jax
assert any(d.platform == "gpu" for d in jax.devices()), "No GPU visible to JAX."
print(f"GPU preflight OK: {jax.devices()}")
PY
[ $? -ne 0 ] && { echo "GPU preflight FAILED."; exit 1; }

stdbuf -o0 -e0 python experiments/run_amoc_hosing.py "$@"
echo "Sweep finished at $(date)."
