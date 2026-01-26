#!/bin/bash
#SBATCH --job-name=test_biharmonic
#SBATCH --output=logs/test_biharmonic_%j.log
#SBATCH --error=logs/test_biharmonic_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --qos=gpushort

# Activate Environment
source venv/bin/activate

# JAX Optimizations
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cuda
export JAX_PLATFORM_NAME=gpu

# Run Control
# kappa_h = 500 (Low, to allow gradients)
# kappa_bi = 1.0e15 (Strong damping for noise)
# T31 Resolution approx dx ~ 3e5 m.
# Stability: dt * kappa_bi / dx^4 < 1/16 ? 
# dt=900. dx=3e5. dx^4 = 81e20 = 8e21.
# 1e15 * 900 / 8e21 ~ 1e-4. Very stable.
# Could go higher? 
# Hyperviscosity typically scales as U*dx^3 ? 0.1 * (3e5)^3 ~ 2.7e15. 
# Let's try 1e15 and maybe 5e15.

python experiments/control_run.py \
    --years 1.0 \
    --r_drag 0.05 \
    --kappa_gm 1000.0 \
    --kappa_h 500.0 \
    --kappa_bi 1.0e15 \
    --suffix prod_v4_test_bi_1e15

