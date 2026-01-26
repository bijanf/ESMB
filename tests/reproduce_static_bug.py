
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np

# Add parent directory to path
sys.path.append(os.getcwd())

from chronos_esm import main, config
from chronos_esm.atmos import dynamics

def check_init():
    print("Initializing Model...")
    state = main.init_model()
    
    # Create mask: Left Half Ocean (1), Right Half Land (0)
    # Use config dimensions to be safe
    ny, nx = config.ATMOS_GRID.nlat, config.ATMOS_GRID.nlon
    ocean_mask = jnp.zeros((ny, nx))
    ocean_mask = ocean_mask.at[:, : int(nx/2)].set(1.0)
    
    u_mean = jnp.mean(state.atmos.u)
    u_max = jnp.max(state.atmos.u)
    vort_max = jnp.max(jnp.abs(state.atmos.vorticity))
    
    print(f"Initial U Mean: {u_mean}")
    print(f"Initial U Max: {u_max}")
    print(f"Initial Vorticity Max: {vort_max}")
    
    if u_max == 0.0:
        print("FAIL: Initial winds are zero!")
        return
        
    print("Running 1 Step...")
    params = main.ModelParams(mask=ocean_mask)
    regridder = main.regrid.Regridder()
    
    next_state = main.step_coupled(state, params, regridder)
    # Force computation
    next_state.atmos.u.block_until_ready()
    
    print("Running 100 Steps...")
    params = main.ModelParams()
    regridder = main.regrid.Regridder()
    
    current_state = next_state
    for i in range(100):
        current_state = main.step_coupled(current_state, params, regridder)
        
    # Force computation
    current_state.atmos.u.block_until_ready()
    
    u_mean_final = jnp.mean(current_state.atmos.u)
    u_max_final = jnp.max(current_state.atmos.u)
    
    print(f"Step 101 U Mean: {u_mean_final}")
    print(f"Step 101 U Max: {u_max_final}")
    
    t_land_initial = 290.0
    t_land_final = jnp.mean(current_state.land.temp)
    print(f"Initial Land Temp: {t_land_initial}")
    print(f"Step 101 Land Temp: {t_land_final}")
    
    if abs(t_land_final - t_land_initial) < 1e-4:
        print("FAIL: Land Temp is constant!")
    
    if u_max_final < 0.01:
        print("FAIL: Winds decayed to near zero!")

if __name__ == "__main__":
    check_init()
