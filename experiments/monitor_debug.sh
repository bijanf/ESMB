#!/bin/bash
echo "Monitoring Debug Suite (including High Diff)..."
while squeue -u fallah | grep -q debug; do
  squeue -u fallah | grep debug
  sleep 30
done
echo "All Debug Jobs Finished. Running Verification..."
source venv/bin/activate
python experiments/verify_debug_suite.py
