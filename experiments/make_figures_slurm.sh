#!/bin/bash
#SBATCH --job-name=esm_figs
#SBATCH --output=logs/esm_figs_%j.log
#SBATCH --error=logs/esm_figs_%j.err
#SBATCH --partition=gpu
#SBATCH --qos=gpushort
#SBATCH --account=poem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Regenerate the validation-dashboard + dino-circulation figures from the latest control run.
#   sbatch experiments/make_figures_slurm.sh /p/tmp/$USER/dino_control_lib "130-yr dino control"
CTRLDIR="${1:-/p/tmp/$USER/dino_control_lib}"
LABEL="${2:-dino control}"

module load python/3.12.3
module load cuda/12.6.0
source venv/bin/activate
export CHRONOS_RESOLUTION=T31
export JAX_PLATFORM_NAME=gpu
export JAX_ENABLE_X64=True
export LD_LIBRARY_PATH=$(python -c "import os; import nvidia; print(':'.join([os.path.join(nvidia.__path__[0], d, 'lib') for d in os.listdir(nvidia.__path__[0]) if os.path.isdir(os.path.join(nvidia.__path__[0], d, 'lib'))]))"):$LD_LIBRARY_PATH

mkdir -p logs docs/figures
echo "Generating figures at $(date); control=$CTRLDIR"

# (1) Coupled-control climatology dashboard (last ~30 annual checkpoints -> time mean).
#     --no-readme: write the figures only; the README is updated separately on the workstation.
stdbuf -o0 -e0 python experiments/make_readme_figures.py \
    "$CTRLDIR"/state_d04[5-7]*.nc --label "$LABEL" --no-readme

# (2) Dino atmospheric circulation (jet, surface winds, ITCZ) -- 90-day spin-up, 30-day mean.
stdbuf -o0 -e0 python experiments/dino_circulation_figure.py \
    --spinup 90 --avg 30 --out docs/figures/dino_circulation.png

echo "Figures done at $(date). Updated docs/figures/:"
ls -lt docs/figures/*.png | head
