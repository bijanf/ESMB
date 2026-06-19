#!/bin/bash
#SBATCH --job-name=dino_ctrl_lib
#SBATCH --output=logs/dino_ctrl_lib_%j.log
#SBATCH --error=logs/dino_ctrl_lib_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Coupled control run on the LIBRARY multi-level stepper (experiments/run_dino_control.py
# -> chronos_esm.coupler.dino_step.DinoCoupledModel: dino atmos + ocean + land + Semtner ice
# as one differentiable, checkpointable step). Also accumulates the windowed q-flux
# (<lambda*(SST_target-SST)>) per checkpoint for the forcing-responsive free run (P2).
#
# Usage:
#   sbatch experiments/run_dino_control_lib_slurm.sh --years 100
#   # chain another job: AUTO-RESUMES from the latest .npz checkpoint:
#   sbatch experiments/run_dino_control_lib_slurm.sh --years 200

module load python/3.12.3
module load cuda/12.6.0
source venv/bin/activate

export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

OUTDIR="${CHRONOS_OUTDIR:-/p/tmp/$USER/dino_control_lib}"
mkdir -p "$OUTDIR" logs
echo "Output dir (scratch): $OUTDIR"
echo "Starting library dino control run at $(date); Job ID: $SLURM_JOB_ID"

# --- GPU preflight ------------------------------------------------------------
python - <<'PY'
import jax
devs = jax.devices()
print(f"JAX devices: {devs}")
assert any(d.platform == "gpu" for d in devs), "No GPU visible to JAX -- check cuda/jax[cuda]."
print("GPU preflight OK")
PY
[ $? -ne 0 ] && { echo "GPU preflight FAILED -- aborting."; exit 1; }

# --- data cache preflight (compute nodes have no internet) ---------------------
python experiments/prefetch_data.py --check
[ $? -ne 0 ] && { echo "Datasets not staged -- run prefetch_data.py on a login node."; exit 1; }

# --- auto-resume from the latest .npz checkpoint unless --resume was given -----
ARGS="$@"
if [[ "$ARGS" != *"--resume"* ]]; then
  latest=$(python -c "import glob,re,os; fs=glob.glob('$OUTDIR/state_d*.npz'); d=[int(m.group(1)) for f in fs for m in [re.search(r'state_d(\d+)\.npz\$',os.path.basename(f))] if m]; print(max(d) if d else '')" 2>/dev/null)
  if [[ -n "$latest" ]]; then
    echo "Auto-resuming from day $latest"
    ARGS="$ARGS --resume $latest"
  else
    echo "No existing checkpoint -- fresh start."
  fi
fi
[[ "$ARGS" != *"--outdir"* ]] && ARGS="$ARGS --outdir $OUTDIR"

stdbuf -o0 -e0 python experiments/run_dino_control.py $ARGS

echo "Run finished at $(date). Checkpoints in: $OUTDIR"
