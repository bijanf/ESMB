#!/bin/bash
# Run a short simulation (1 month) to check noise
# Fixed JAX flags for head node
export JAX_PLATFORMS=cpu
source venv/bin/activate
export PYTHONPATH=$(pwd)

echo "Running verification run with kappa_h=100.0..."
python experiments/control_run.py --years 0.08333 --kappa_h 100.0 --suffix verify_noise
