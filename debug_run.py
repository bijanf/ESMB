
import sys
import os
sys.path.insert(0, os.getcwd())

import jax
from chronos_esm import main, data

def run_debug():
    print("Initializing Model...")
    # Ensure we load the REAL mask
    mask = data.load_bathymetry_mask(nz=15)
    params = main.ModelParams(mask=mask)
    
    print("Running Simulation for 2 steps...")
    # Run just a few steps to trigger prints
    # Note: step_coupled is scanned, so prints might happen during JIT or Execution
    # With jax.debug.print, they happen during Execution.
    final_state = main.run_simulation(steps=2, params=params)
    
    print("Debug Run Complete.")

if __name__ == "__main__":
    try:
        run_debug()
    except Exception as e:
        print(f"Run failed: {e}")
