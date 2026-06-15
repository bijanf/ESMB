#!/bin/bash
#SBATCH --job-name=dino_control
#SBATCH --output=logs/dino_control_%j.log
#SBATCH --error=logs/dino_control_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpumedium
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=23:00:00

# Coupled multi-level (dinosaur) atmosphere <-> ocean CONTROL RUN.
#
# Usage:
#   # fresh 100-year run, checkpoint yearly:
#   sbatch experiments/run_dino_control_slurm.sh --years 100
#   # chain another 23h job: it AUTO-RESUMES from the latest checkpoint:
#   sbatch experiments/run_dino_control_slurm.sh --years 200
#   # explicit resume (overrides auto):
#   sbatch experiments/run_dino_control_slurm.sh --years 200 --resume 36500

# Load necessary modules
module load anaconda
module load cuda/12.3.1

# Activate Virtual Environment
# NOTE: this needs a GPU JAX build (requirements.txt installs CPU JAX). One-time setup:
#   pip install "jax[cuda12]==$(python -c 'import jax; print(jax.__version__)')"
# Otherwise the LD_LIBRARY_PATH (nvidia wheels) line and the GPU preflight below fail.
source venv/bin/activate

# Resolution / JAX platform
export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True

# NVIDIA Libs
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

OUTDIR=outputs/dino_control
mkdir -p "$OUTDIR" logs

echo "Starting Chronos-ESM dinosaur<->ocean control run at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# --- GPU preflight: confirm JAX sees the GPU before a multi-hour job ----------
python - <<'PY'
import jax
devs = jax.devices()
print(f"JAX devices: {devs}")
assert any(d.platform == "gpu" for d in devs), \
    "No GPU visible to JAX -- check cuda module / jax[cuda] install before a long run."
print("GPU preflight OK")
PY
if [ $? -ne 0 ]; then
  echo "GPU preflight FAILED -- aborting." ; exit 1
fi

# --- data cache preflight: compute nodes have no internet, so WOA18 + ETOPO1
# must already be staged in ~/.cache/chronos_esm. Stage them on a LOGIN node with:
#   python experiments/prefetch_data.py
python experiments/prefetch_data.py --check
if [ $? -ne 0 ]; then
  echo "Datasets not staged -- run 'python experiments/prefetch_data.py' on a login node first." ; exit 1
fi

# --- auto-resume from the latest checkpoint unless --resume was given ---------
ARGS="$@"
if [[ "$ARGS" != *"--resume"* ]]; then
  latest=$(python -c "import glob,re,os; fs=glob.glob('$OUTDIR/state_d*.nc'); d=[int(re.search(r'state_d(\d+)\.nc',os.path.basename(f)).group(1)) for f in fs]; print(max(d) if d else '')" 2>/dev/null)
  if [[ -n "$latest" ]]; then
    echo "Auto-resuming from day $latest"
    ARGS="$ARGS --resume $latest"
  else
    echo "No existing checkpoint -- fresh start."
  fi
fi

stdbuf -o0 -e0 python experiments/run_dino_coupled.py $ARGS

echo "Run finished at $(date)"
