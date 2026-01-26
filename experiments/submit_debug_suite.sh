#!/bin/bash

# 1. Baseline (kappa_h=1000)
sbatch --job-name=debug_base --output=logs/debug_base_%j.log experiments/run_debug_job.sh 0.25 1000.0 debug_base

# 2. Moderate (kappa_h=3000)
sbatch --job-name=debug_diff3k --output=logs/debug_diff3k_%j.log experiments/run_debug_job.sh 0.25 3000.0 debug_diff3k

# 3. High (kappa_h=5000)
sbatch --job-name=debug_diff5k --output=logs/debug_diff5k_%j.log experiments/run_debug_job.sh 0.25 5000.0 debug_diff5k
