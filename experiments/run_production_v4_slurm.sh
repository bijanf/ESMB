#!/bin/bash
#SBATCH --job-name=prod_v4
#SBATCH --output=logs/prod_v4_%j.log
#SBATCH --error=logs/prod_v4_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --qos=gpushort

# Activate Environment
source venv/bin/activate

# JAX Optimizations
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cuda
export JAX_PLATFORM_NAME=gpu

# Production Run V4
# Physics: Biharmonic Mixing for Noise Control
# kappa_h = 500 (Low Laplacian to preserve gradients)
# kappa_bi = 1.0e15 (Biharmonic to kill grid noise)
# kappa_gm = 1000 (Eddy mixing)
# r_drag = 0.05 (Friction)

python experiments/control_run.py \
    --years 100.0 \
    --r_drag 0.05 \
    --kappa_gm 1000.0 \
    --kappa_h 500.0 \
    --kappa_bi 1.0e15 \
    --Ah 0.0 \
    --Ab 0.0 \
    --shapiro_strength 0.0 \
    --smag_constant 0.0 \
    --suffix prod_v29
