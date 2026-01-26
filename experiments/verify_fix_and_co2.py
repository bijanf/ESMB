
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import data, main
from chronos_esm.coupler import state as coupled_state

def run_verification():
    print("Setting up verification run...")
    
    # 1. Setup Model with High CO2 Increase
    regridder = main.regrid.Regridder()
    mask = data.load_bathymetry_mask()
    
    # Increase CO2 by 50% per year to see effect clearly in 1 month
    # 1.5^(1/12) ~= 1.034 (+3.4%)
    params = main.ModelParams(
        mask=mask,
        co2_ppm=280.0,
        co2_increase_rate=0.5 # 50%
    )
    
    # 2. Run for 1 month (approx 43200 steps at dt=60, or fewer if we want speed)
    # Let's run 1000 steps (~16 hours) to look for immediate crash, then check CO2
    steps = 2880 # 1 day at dt=30
    
    print(f"Running for {steps} steps...")
    
    final_state = main.run_simulation(steps, params)
    final_state.ocean.temp.block_until_ready()
    
    # 3. Check CO2
    final_co2 = float(jnp.mean(final_state.atmos.co2))
    print(f"Final Mean CO2: {final_co2:.2f} ppm")
    
    from chronos_esm.config import DT_ATMOS
    expected_co2 = 280.0 * (1.0 + 0.5)**(steps * DT_ATMOS / (365*24*3600))
    print(f"Expected CO2: {expected_co2:.2f} ppm")
    
    # 4. Check Stability
    t_mean = float(jnp.mean(final_state.atmos.temp))
    t_min = float(jnp.min(final_state.atmos.temp))
    t_max = float(jnp.max(final_state.atmos.temp))
    
    print(f"Final T_mean: {t_mean:.2f} K")
    print(f"Final T_range: [{t_min:.2f}, {t_max:.2f}] K")
    
    if np.isnan(t_mean) or t_mean > 350 or t_mean < 150:
        print("[FAIL] Model Instability Detected!")
    else:
        print("[PASS] Model appears stable (short term).")
        
    if abs(final_co2 - expected_co2) > 0.1:
         print("[WARN] CO2 mismatch!")
    else:
         print("[PASS] CO2 forcing working as expected.")

if __name__ == "__main__":
    run_verification()
