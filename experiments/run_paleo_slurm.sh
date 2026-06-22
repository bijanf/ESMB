#!/bin/bash
#SBATCH --job-name=paleo_mh
#SBATCH --output=logs/paleo_mh_%j.log
#SBATCH --error=logs/paleo_mh_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Free-mode seasonal mid-Holocene paleo run (experiments/run_paleo_midholocene.py).
# All args are passed through. Use the SAME --ckpt + q-flux (from the seasonal PI control)
# for both orbits so only the orbital config differs (PMIP-style).
#
# Usage (launch both orbits with the same seasonal-control checkpoint):
#   CK=/p/tmp/$USER/seasonal_control_pi/state_d021900
#   sbatch experiments/run_paleo_slurm.sh --ckpt $CK --orbit pi  --years 40 --outdir outputs/paleo_pi
#   sbatch experiments/run_paleo_slurm.sh --ckpt $CK --orbit 6ka --years 40 --outdir outputs/paleo_6ka

module load python/3.12.3
module load cuda/12.6.0
source venv/bin/activate

export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

mkdir -p logs
echo "Starting mid-Holocene paleo run at $(date); Job ID: $SLURM_JOB_ID"
echo "Args: $@"

# --- GPU preflight ------------------------------------------------------------
python - <<'PY'
import jax
devs = jax.devices()
print(f"JAX devices: {devs}")
assert any(d.platform == "gpu" for d in devs), "No GPU visible to JAX -- check cuda/jax[cuda]."
print("GPU preflight OK")
PY
[ $? -ne 0 ] && { echo "GPU preflight FAILED -- aborting."; exit 1; }

python experiments/prefetch_data.py --check
[ $? -ne 0 ] && { echo "Datasets not staged -- run prefetch_data.py on a login node."; exit 1; }

# auto-add --resume if a state checkpoint for this orbit already exists in --outdir
ARGS="$@"
OUTDIR=$(echo "$ARGS" | sed -n 's/.*--outdir \([^ ]*\).*/\1/p')
ORBIT=$(echo "$ARGS" | sed -n 's/.*--orbit \([^ ]*\).*/\1/p'); ORBIT=${ORBIT:-pi}
if [[ -n "$OUTDIR" && -f "$OUTDIR/state_${ORBIT}.npz" && "$ARGS" != *"--resume"* ]]; then
  echo "Found $OUTDIR/state_${ORBIT}.npz -- adding --resume"
  ARGS="$ARGS --resume"
fi

stdbuf -o0 -e0 python experiments/run_paleo_midholocene.py $ARGS

echo "Paleo run finished at $(date)."
