#!/bin/bash
echo "Monitoring Debug Suite (including High Diff)..."
while squeue -u "$USER" | grep -q debug; do
  squeue -u "$USER" | grep debug
  sleep 30
done
echo "All Debug Jobs Finished. Running Verification..."
source venv/bin/activate
python experiments/verify_debug_suite.py
