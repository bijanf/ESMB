
import os
import shutil
import jax
import jax.numpy as jnp
import sys
sys.path.append(os.getcwd())
try:
    from chronos_esm.config import OCEAN_GRID, DT_OCEAN
except ImportError:
    sys.path.append("/home/fallah/scripts/ESMB")
    from chronos_esm.config import OCEAN_GRID, DT_OCEAN
from chronos_esm.ocean.veros_driver import step_ocean, init_ocean_state, OceanState
from chronos_esm.ocean import solver

# Mock State
nz, ny, nx = OCEAN_GRID.nz, OCEAN_GRID.nlat, OCEAN_GRID.nlon
dx = 2 * jnp.pi * 6371000 / nx # approx
dy = jnp.pi * 6371000 / ny
dz = jnp.ones(nz) * 50.0

state = init_ocean_state(nz, ny, nx)

# Initialize with a hot spot near pole to see if it diffuses or explodes
temp = state.temp
# Add anomaly at North Pole
temp = temp.at[:, -5:, :].set(300.0) # Hot north pole
state = state._replace(temp=temp)

# Fluxes zero
fluxes = (jnp.zeros((ny, nx)), jnp.zeros((ny, nx)), jnp.zeros((ny, nx)))
stress = (jnp.zeros((ny, nx)), jnp.zeros((ny, nx)))

print("Running Polar Stability Test (100 steps)...")
print(f"Initial Max Temp: {jnp.max(state.temp):.4f} K")
print(f"Initial North Pole Mean: {jnp.mean(state.temp[:, -1, :]):.4f} K")

kappa_bi = 1e15
kappa_h = 500.0

for i in range(100):
    state = step_ocean(state, fluxes, stress, dx, dy, dz, kappa_bi=kappa_bi, kappa_h=kappa_h)
    
print(f"Final Max Temp: {jnp.max(state.temp):.4f} K")
print(f"Final North Pole Mean: {jnp.mean(state.temp[:, -1, :]):.4f} K")

# Check if it exploded or cooled (Sponge should cool it or keep it stable)
# Initial was 300.
if jnp.max(state.temp) > 301.0:
    print(f"FAILED: Instability detected! Max: {jnp.max(state.temp)}")
else:
    print("PASSED: Stable diffusion.")
