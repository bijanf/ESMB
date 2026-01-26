
import sys
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from chronos_esm import main, data
from chronos_esm import io as model_io
from chronos_esm.config import DT_OCEAN, DT_ATMOS

def verify_run():
    print(f"Starting Verification Run...")
    print(f"DT_ATMOS: {DT_ATMOS}, DT_OCEAN: {DT_OCEAN}")
    
    # Initialize
    state = main.init_model()
    regridder = main.regrid.Regridder()
    params = main.ModelParams()
    
    # Run for 1 Month
    # 1 Month = 30 * 24 * 3600 seconds
    seconds_per_month = 30 * 24 * 3600
    steps = int(seconds_per_month / DT_OCEAN)
    
    print(f"Running for {steps} coupled steps (1 month)...")
    
    start_time = time.time()
    
    # Compile scan
    def step_fn(carry, _):
        s = carry
        new_s = main.step_coupled(s, params, regridder)
        return new_s, None
        
    final_state, _ = jax.lax.scan(step_fn, state, jnp.arange(steps))
    
    # Block
    final_state.ocean.temp.block_until_ready()
    end_time = time.time()
    
    print(f"Run complete in {end_time - start_time:.2f}s")
    
    # Check Stability
    t_min = jnp.min(final_state.atmos.temp)
    t_max = jnp.max(final_state.atmos.temp)
    u_max = jnp.max(jnp.abs(final_state.atmos.u))
    v_max = jnp.max(jnp.abs(final_state.atmos.v))
    
    print(f"Atmos Temp Range: {t_min:.2f}K - {t_max:.2f}K")
    print(f"Max Wind Speed: {u_max:.2f} m/s (U), {v_max:.2f} m/s (V)")
    
    if t_min < 150.0 or t_max > 350.0:
        print("[FAILED] Temperature out of bounds!")
        sys.exit(1)
        
    if u_max > 150.0 or v_max > 150.0:
        print("[FAILED] Winds exploded!")
        sys.exit(1)
        
    # Check Noise (Laplacian on SST)
    # SST is in fluxes.sst (K)
    sst = final_state.fluxes.sst
    
    # Numpy for rolling
    sst_np = np.array(sst)
    
    sst_roll_xm = np.roll(sst_np, 1, axis=1)
    sst_roll_xp = np.roll(sst_np, -1, axis=1)
    sst_roll_ym = np.roll(sst_np, 1, axis=0)
    sst_roll_yp = np.roll(sst_np, -1, axis=0)
    
    laplacian = np.abs(sst_roll_xp + sst_roll_xm + sst_roll_yp + sst_roll_ym - 4*sst_np)
    
    max_noise = np.max(laplacian)
    mean_noise = np.mean(laplacian)
    
    print(f"SST Laplacian Noise - Max: {max_noise:.4f} K, Mean: {mean_noise:.4f} K")
    
    if max_noise > 10.0:
         print("[WARNING] High grid-scale noise detected!")
    else:
         print("[PASSED] Noise levels acceptable.")

if __name__ == "__main__":
    verify_run()
