
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import main, data
from chronos_esm.ocean import diagnostics

# Disable JIT for maximum inspectability? No, too slow.
# Use block_until_ready inside loop.

def run_trace():
    print("Running Trace Instability Diagnostic...")
    
    # Setup
    mask = jnp.array(data.load_bathymetry_mask())
    params = main.ModelParams(mask=mask, co2_ppm=280.0)
    state = main.init_model()
    regridder = main.regrid.Regridder()
    
    # 1. Spinup
    print("\n--- Phase 1: Spinup (Static) ---")
    defaults_r = 1.0e-2
    spinups = 350
    
    @jax.jit
    def spinup(s):
        s, _ = jax.lax.scan(lambda c, _: (main.step_coupled(c, params, regridder, r_drag=defaults_r), None), s, None, length=spinups)
        return s
        
    print(f"Running {spinups} steps of spinup...")
    state = spinup(state)
    state.ocean.temp.block_until_ready()
    print("Spinup Complete.")
    
    # 2. Forward Simulation Trace
    print("\n--- Phase 2: Trace (Day 0 to Day 10) ---")
    # Tuning crashed at Step 18 * 24 = 432.
    # We run 500 steps.
    # Each step = 900s (15 mins).
    # 24 steps = 6 hours.
    
    current_r = 0.01010 # Roughly where it crashed
    print(f"Tracing with r_drag = {current_r}")
    
    @jax.jit
    def single_step(s):
        return main.step_coupled(s, params, regridder, r_drag=current_r)
        
    for i in range(500):
        # Run 1 step
        state = single_step(state)
        
        # Block and Check
        state.ocean.u.block_until_ready()
        
        # Check specific fields
        u_max = jnp.max(jnp.abs(state.ocean.u))
        v_max = jnp.max(jnp.abs(state.ocean.v))
        temp_mean = jnp.mean(state.ocean.temp)
        
        # Pull to host
        u_m = float(u_max)
        v_m = float(v_max)
        t_m = float(temp_mean)
        
        if np.isnan(u_m) or np.isnan(v_m) or np.isnan(t_m):
            print(f"STEP {i}: CRASH DETECTED (NaN)")
            # Diagnostics?
            break
            
        if np.isinf(u_m) or np.isinf(v_m):
            print(f"STEP {i}: CRASH DETECTED (Inf)")
            break
            
        if i % 24 == 0:
            print(f"Step {i:<4} (Day {i//96:.1f}): U_max={u_m:.2e}, V_max={v_m:.2e}, T_mean={t_m:.2f}")

    print("Trace Complete.")

if __name__ == "__main__":
    run_trace()
