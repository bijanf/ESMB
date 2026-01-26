
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add project root
sys.path.append(os.getcwd())

from chronos_esm.ocean import veros_driver
from chronos_esm.config import OCEAN_GRID, RHO_WATER, DT_OCEAN

def test_convection():
    print("Testing Convective Adjustment...")
    
    # Setup 1D column (nz, 1, 1) masked as (nz, 3, 3) to satisfy stencils
    nz = 15
    ny = 4
    nx = 4
    dz = jnp.ones(nz) * 100.0
    dx = 1e5
    dy = 1e5
    
    # Initial State: Unstable
    # Cold on top (10C), Warm below (20C)
    temp = jnp.ones((nz, ny, nx)) * 20.0
    temp = temp.at[0, :, :].set(10.0) # Surface cold
    
    salt = jnp.ones((nz, ny, nx)) * 35.0
    
    state = veros_driver.OceanState(
        u=jnp.zeros((nz, ny, nx)),
        v=jnp.zeros((nz, ny, nx)),
        w=jnp.zeros((nz, ny, nx)),
        temp=temp + 273.15,
        salt=salt,
        psi=jnp.zeros((ny, nx)),
        rho=jnp.zeros((nz, ny, nx)), # Will be updated
        dic=jnp.zeros((nz, ny, nx))
    )
    
    # Zero forcings
    fluxes = (jnp.zeros((ny, nx)), jnp.zeros((ny, nx)), jnp.zeros((ny, nx)))
    stress = (jnp.zeros((ny, nx)), jnp.zeros((ny, nx)))
    
    # Run for a few steps
    # time scale for mixing 100m with K=1.0 is T ~ L^2/K ~ 10000s ~ 10 steps
    # We run 20 steps
    
    print(f"Initial Surface T: {state.temp[0,1,1]-273.15:.2f} C")
    print(f"Initial Deep T: {state.temp[1,1,1]-273.15:.2f} C")
    
    # JIT the step
    step_fn = veros_driver.step_ocean
    
    for i in range(50):
        state = step_fn(
            state, fluxes, stress, 
            dx=dx, dy=dy, dz=dz, 
            nz=nz, ny=ny, nx=nx,
            kappa_gm=0.0, kappa_h=0.0 # pure vertical physics check
        )
        if i % 10 == 0:
            print(f"Step {i}: Surface T={state.temp[0,1,1]-273.15:.2f} C")

    t_surf = state.temp[0,1,1]-273.15
    t_deep = state.temp[1,1,1]-273.15
    
    print(f"Final Surface T: {t_surf:.2f} C")
    print(f"Final Deep T: {t_deep:.2f} C")
    
    # valid convection should warm the surface and cool the deep
    if t_surf > 10.1 and t_deep < 19.9:
        print("PASS: Convection mixed the column.")
    else:
        print("FAIL: No significant mixing observed.")
        
    # Check Stability
    rho = veros_driver.equation_of_state(state.temp, state.salt)
    drho = rho[1:, 1, 1] - rho[:-1, 1, 1]
    # Should be positive (stable)
    if jnp.all(drho > -0.01):
        print("PASS: Column is stable.")
    else:
        print(f"FAIL: Column unstable. Min drho: {jnp.min(drho)}")

if __name__ == "__main__":
    test_convection()
