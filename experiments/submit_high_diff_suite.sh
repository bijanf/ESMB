#!/bin/bash

# 1. High Diffusivity 15k
sbatch --job-name=debug_kh15k --output=logs/debug_kh15k_%j.log experiments/run_debug_job.sh 0.25 15000.0 debug_kh15k

# 2. High Diffusivity 20k
sbatch --job-name=debug_kh20k --output=logs/debug_kh20k_%j.log experiments/run_debug_job.sh 0.25 20000.0 debug_kh20k
