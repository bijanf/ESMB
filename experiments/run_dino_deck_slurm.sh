#!/bin/bash
#SBATCH --job-name=dino_deck
#SBATCH --output=logs/dino_deck_%j.log
#SBATCH --error=logs/dino_deck_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpumedium
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=23:00:00

# CMIP-DECK trio in ONE job, run AFTER a converged library control run.
#
# Runs piControl + 1pctCO2 + abrupt-4xCO2 (the CMIP-shaped idealized set) on the
# freed (q-flux) coupled model, resuming from the latest control checkpoint written
# by run_dino_control.py / run_dino_control_lib_slurm.sh (state_d<DAY>.npz +
# _dino.npz + _qflux.npy), then runs the CMIP6/AR6 comparison. All output to SCRATCH.
#
# Prereqs (do these first, in order):
#   1) python experiments/prefetch_data.py            # stage WOA18+ETOPO1 on a LOGIN node
#   2) sbatch experiments/run_dino_control_lib_slurm.sh --years 100   # the control run
#   3) sbatch --requeue experiments/run_dino_deck_slurm.sh            # this script
#
# Knobs (env):
#   CHRONOS_CTRLDIR  control checkpoint dir   (default /p/tmp/$USER/dino_control_lib)
#   CHRONOS_OUTDIR   DECK output dir (scratch)(default /p/tmp/$USER/dino_deck)
#   DECK_EXPERIMENTS branches to run          (default "piControl 1pctCO2 abrupt-4xCO2")
#   DECK_YEARS       length of each branch    (default 150)
# Any extra CLI args are forwarded to every run_dino_deck.py call (e.g. --interval 1).
#
# Idempotent / requeue-safe: a branch whose deck_<exp>.npz already exists is SKIPPED,
# so a requeued 23 h job resumes the trio where it left off.

module load python/3.12.3
module load cuda/12.6.0
source venv/bin/activate

export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

CTRLDIR="${CHRONOS_CTRLDIR:-/p/tmp/$USER/dino_control_lib}"
OUTDIR="${CHRONOS_OUTDIR:-/p/tmp/$USER/dino_deck}"
EXPERIMENTS="${DECK_EXPERIMENTS:-piControl 1pctCO2 abrupt-4xCO2}"
YEARS="${DECK_YEARS:-150}"
mkdir -p "$OUTDIR" logs
echo "Control dir: $CTRLDIR"
echo "DECK output dir (scratch): $OUTDIR"
echo "Experiments: $EXPERIMENTS  (years=$YEARS each)"
echo "Starting DECK trio at $(date); Job ID: $SLURM_JOB_ID"

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

# --- locate the latest control checkpoint (state_d<DAY>.npz, excluding _dino) ---
latest=$(python -c "import glob,re,os; fs=glob.glob('$CTRLDIR/state_d*.npz'); d=[int(m.group(1)) for f in fs for m in [re.search(r'state_d(\d+)\.npz\$',os.path.basename(f))] if m]; print(max(d) if d else '')" 2>/dev/null)
if [[ -z "$latest" ]]; then
  echo "No control checkpoint (state_d*.npz) in $CTRLDIR."
  echo "Run the control first: sbatch experiments/run_dino_control_lib_slurm.sh --years 100"
  exit 1
fi
CKPT="$CTRLDIR/state_d$(printf '%06d' "$latest")"
echo "Resuming DECK from control checkpoint: $CKPT (day $latest)"

# the DECK driver needs the frozen q-flux written alongside the checkpoint
if [[ ! -f "${CKPT}_qflux.npy" ]]; then
  echo "Missing ${CKPT}_qflux.npy -- the control run must write the windowed q-flux."
  echo "(run_dino_control.py saves it per checkpoint; check the control run completed a window.)"
  exit 1
fi

# --- run each branch (skip if already finished -> requeue-safe) ----------------
rc_all=0
for exp in $EXPERIMENTS; do
  out="$OUTDIR/deck_${exp}.npz"
  if [[ -f "$out" ]]; then
    echo "[$exp] already done ($out) -- skipping."
    continue
  fi
  echo "=== [$exp] $(date) ==="
  stdbuf -o0 -e0 python experiments/run_dino_deck.py \
    --experiment "$exp" --ckpt "$CKPT" --years "$YEARS" --outdir "$OUTDIR" "$@"
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[$exp] FAILED (rc=$rc)."
    rc_all=$rc
  fi
done

# --- CMIP6/AR6 comparison (figure to scratch, NOT the repo/home) ---------------
if [[ -f "$OUTDIR/deck_piControl.npz" ]]; then
  echo "=== comparing to CMIP6/AR6 $(date) ==="
  python experiments/compare_cmip6_deck.py \
    --deckdir "$OUTDIR" --out "$OUTDIR/cmip6_deck_comparison.pdf" || true
fi

echo "DECK trio finished at $(date). Outputs in: $OUTDIR"
exit $rc_all
