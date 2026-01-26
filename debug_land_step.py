
import sys
import os
sys.path.insert(0, os.getcwd())

import jax
import jax.numpy as jnp
from chronos_esm.land import driver as land_driver
from chronos_esm.config import ATMOS_GRID

def debug_land():
    ny, nx = 48, 96
    state = land_driver.init_land_state(ny, nx)
    
    # Inputs
    t_air = jnp.ones((ny, nx)) * 300.0
    q_air = jnp.ones((ny, nx)) * 0.01
    sw_down = jnp.ones((ny, nx)) * 300.0
    lw_down = jnp.ones((ny, nx)) * 300.0
    precip = jnp.zeros((ny, nx))
    
    # Mask: Half land, Half ocean
    mask = jnp.zeros((ny, nx))
    mask = mask.at[:, :nx//2].set(1.0) # Left half Land (1.0)
    
    wind_speed = jnp.ones((ny, nx)) * 10.0
    drag_coeff = jnp.ones((ny, nx)) * 0.002
    
    # Run step
    print("Running step_land...")
    new_state, fluxes = land_driver.step_land(
        state,
        t_air=t_air,
        q_air=q_air,
        sw_down=sw_down,
        lw_down=lw_down,
        precip=precip,
        mask=mask,
        wind_speed=wind_speed,
        drag_coeff=drag_coeff
    )
    
    # Check Temp
    t_old = state.temp
    t_new = new_state.temp
    
    diff = t_new - t_old
    
    print(f"Old Temp Mean: {jnp.mean(t_old)}")
    print(f"New Temp Mean: {jnp.mean(t_new)}")
    print(f"Max Diff: {jnp.max(diff)}")
    print(f"Min Diff: {jnp.min(diff)}")
    
    # Check if masked area updated
    land_diff = jnp.mean(diff[:, :nx//2])
    ocean_diff = jnp.mean(diff[:, nx//2:])
    
    print(f"Land (Mask=1) Mean Diff: {land_diff}")
    print(f"Ocean (Mask=0) Mean Diff: {ocean_diff}")
    
    if jnp.abs(land_diff) > 1e-6:
        print("SUCCESS: Land temperature updated.")
    else:
        print("FAILURE: Land temperature DID NOT update.")

if __name__ == "__main__":
    debug_land()
