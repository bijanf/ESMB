
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from chronos_esm.ocean import veros_driver
from chronos_esm.config import OCEAN_GRID

def test_baroclinic_physics():
    print("Initializing Ocean State with Thermal Gradient...")
    
    nz = OCEAN_GRID.nz
    ny = OCEAN_GRID.nlat
    nx = OCEAN_GRID.nlon
    
    # Create State
    # T = T0 + y * dT/dy
    # Cooler at North (y=ny-1), Warmer at South (y=0)
    # y index 0 is -90, y index ny is 90.
    
    y = jnp.linspace(-90, 90, ny)
    x = jnp.linspace(0, 360, nx)
    
    # Cosine profile for temp: Warm equator, cold poles
    # Add Zonal variation to induce dp/dx -> v_geo
    temp_y = 20.0 * jnp.cos(jnp.deg2rad(y)) 
    temp_x = 5.0 * jnp.sin(jnp.deg2rad(x))
    
    temp_surf = temp_y[:, None] + temp_x[None, :]
    temp_surf = temp_surf + 273.15 # Kelvin
    
    # Broadcast to 3D (nz, ny, nx)
    temp = jnp.broadcast_to(temp_surf[None, :, :], (nz, ny, nx))
    
    # Add vertical decay to make it interesting (baroclinic)
    # T decreases with depth
    z_decay = jnp.linspace(1.0, 0.0, nz)
    temp = temp * z_decay[:, None, None]
    
    state = veros_driver.init_ocean_state(nz, ny, nx)
    state = state._replace(temp=temp)
    
    # Fluxes (Zero forcing to isolate geostrophic adjustment)
    surface_fluxes = (jnp.zeros((ny, nx)), jnp.zeros((ny, nx)), jnp.zeros((ny, nx)))
    wind_stress = (jnp.zeros((ny, nx)), jnp.zeros((ny, nx)))
    
    # Grid
    dx = 100e3
    dy = 100e3
    dz = jnp.ones(nz) * 100.0
    
    print("Stepping Ocean Model...")
    # Compile and Step
    new_state = veros_driver.step_ocean(
        state, 
        surface_fluxes, 
        wind_stress, 
        dx=dx, dy=dy, dz=dz,
        nz=nz, ny=ny, nx=nx,
        dt=3600.0 # 1 hour
    )
    
    new_state.u.block_until_ready()
    
    # Check Velocity Structure
    u = new_state.u
    v = new_state.v
    
    print(f"U shape: {u.shape}")
    
    # Check Standard Deviation across depth
    u_std = jnp.std(u, axis=0)
    v_std = jnp.std(v, axis=0)
    
    max_u_std = jnp.max(u_std)
    max_v_std = jnp.max(v_std)
    
    print(f"Max Std Dev U (Depth): {max_u_std}")
    print(f"Max Std Dev V (Depth): {max_v_std}")
    
    if max_u_std < 1e-10:
        print("FAIL: Velocity is depth-independent (Barotropic only).")
    else:
        print("SUCCESS: Velocity varies with depth (Baroclinic).")
        
    # Check AMOC
    # AMOC logic from manual calculation
    # V is (nz, ny, nx)
    # Zonal sum
    v_zonal = jnp.sum(v * dx, axis=2) # (nz, ny)
    # Vertical integral from top (approx)
    moc = -jnp.cumsum(v_zonal * 100.0, axis=0)
    max_moc = jnp.max(jnp.abs(moc)) / 1e6 # Sv
    
    print(f"Max AMOC Strength: {max_moc} Sv")
    
    if max_moc > 0.01:
        print("SUCCESS: Non-zero AMOC detected.")
    else:
        print("FAIL: AMOC is near zero.")

if __name__ == "__main__":
    test_baroclinic_physics()
