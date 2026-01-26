#!/bin/bash
#SBATCH --job-name=tune_v3
#SBATCH --output=logs/tune_v3_%j.log
#SBATCH --error=logs/tune_v3_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --qos=gpushort

# Activate Environment
source venv/bin/activate

# JAX Optimizations
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cuda
export JAX_PLATFORM_NAME=gpu

# Run Multi-Parameter Auto-Tuning
python experiments/tune_physics_v2.py
