
import jax
import jax.numpy as jnp
import time
import numpy as np
from chronos_esm import main
from chronos_esm.config import DT_ATMOS, ATMOS_GRID, OCEAN_GRID
from chronos_esm.ocean import veros_driver as ocean_driver
from chronos_esm.atmos import dynamics as atmos_driver
from chronos_esm.ice import driver as ice_driver
from chronos_esm.land import driver as land_driver
from chronos_esm.coupler import regrid

def profile_components(steps: int = 10):
    """
    Profile each component individually.
    """
    print(f"Profiling Chronos-ESM components for {steps} steps...")
    
    # Initialize
    print("Initializing state...")
    state = main.init_model()
    regridder = regrid.Regridder()
    params = main.ModelParams()
    
    # Force compilation of init
    state.atmos.temp.block_until_ready()
    
    # Extract components
    ocean = state.ocean
    atmos = state.atmos
    land = state.land
    ice = state.ice
    fluxes = state.fluxes
    
    # Dummy inputs for component stepping
    # Ocean inputs
    fluxes_ocean = (
        jnp.zeros_like(ocean.temp[0]), # Heat
        jnp.zeros_like(ocean.temp[0]), # FW
        jnp.zeros_like(ocean.temp[0])  # Carbon
    )
    wind_ocean = (
        jnp.zeros_like(ocean.u[0]),
        jnp.zeros_like(ocean.v[0])
    )
    dx_ocn = 100e3
    dy_ocn = 100e3
    dz_ocn = jnp.ones(ocean.u.shape[0]) * 100.0
    
    # Atmos inputs
    sfc_temp = jnp.ones_like(atmos.temp) * 300.0
    flux_sens = jnp.zeros_like(atmos.temp)
    flux_lat = jnp.zeros_like(atmos.temp)
    flux_co2 = jnp.zeros_like(atmos.temp)
    
    # Land inputs
    t_air = jnp.ones_like(land.temp) * 290.0
    q_air = jnp.ones_like(land.temp) * 0.01
    sw_down = jnp.ones_like(land.temp) * 200.0
    lw_down = jnp.ones_like(land.temp) * 300.0
    precip = jnp.zeros_like(land.temp)
    mask_land = jnp.ones_like(land.temp)
    
    # Ice inputs
    t_air_ocn = jnp.ones_like(ice.surface_temp) * 270.0
    sw_down_ocn = jnp.zeros_like(ice.surface_temp)
    lw_down_ocn = jnp.ones_like(ice.surface_temp) * 200.0
    ocean_temp = jnp.ones_like(ice.surface_temp) * 271.0
    
    # JIT Compile individual steps
    print("JIT Compiling components...")
    
    @jax.jit
    def step_ocean_fn(o):
        return ocean_driver.step_ocean(
            o, fluxes_ocean, wind_ocean, dx_ocn, dy_ocn, dz_ocn, 
            ocean.u.shape[0], OCEAN_GRID.nlat, OCEAN_GRID.nlon, None, DT_ATMOS
        )
        
    @jax.jit
    def step_atmos_fn(a):
        return atmos_driver.step_atmos(
            a, sfc_temp, flux_sens, flux_lat, flux_co2, DT_ATMOS, ATMOS_GRID.nlat, ATMOS_GRID.nlon
        )
        
    @jax.jit
    def step_land_fn(l):
        return land_driver.step_land(
            l, t_air, q_air, sw_down, lw_down, precip, mask_land, ATMOS_GRID.nlat, ATMOS_GRID.nlon
        )
        
    @jax.jit
    def step_ice_fn(i):
        return ice_driver.step_ice(
            i, t_air_ocn, sw_down_ocn, lw_down_ocn, ocean_temp, OCEAN_GRID.nlat, OCEAN_GRID.nlon, None
        )
        
    # Warmup
    print("Warmup...")
    _ = step_ocean_fn(ocean).u.block_until_ready()
    _ = step_atmos_fn(atmos)[0].temp.block_until_ready()
    _ = step_land_fn(land)[0].temp.block_until_ready()
    _ = step_ice_fn(ice)[0].surface_temp.block_until_ready()
    
    # Timing
    print("Profiling...")
    
    # Ocean
    t0 = time.time()
    for _ in range(steps):
        ocean = step_ocean_fn(ocean)
        ocean.u.block_until_ready()
    t_ocean = (time.time() - t0) / steps
    print(f"Ocean: {t_ocean*1000:.2f} ms/step")
    
    # Atmos
    t0 = time.time()
    for _ in range(steps):
        res = step_atmos_fn(atmos)
        atmos = res[0]
        atmos.temp.block_until_ready()
    t_atmos = (time.time() - t0) / steps
    print(f"Atmos: {t_atmos*1000:.2f} ms/step")
    
    # Land
    t0 = time.time()
    for _ in range(steps):
        res = step_land_fn(land)
        land = res[0]
        land.temp.block_until_ready()
    t_land = (time.time() - t0) / steps
    print(f"Land:  {t_land*1000:.2f} ms/step")
    
    # Ice
    t0 = time.time()
    for _ in range(steps):
        res = step_ice_fn(ice)
        ice = res[0]
        ice.surface_temp.block_until_ready()
    t_ice = (time.time() - t0) / steps
    print(f"Ice:   {t_ice*1000:.2f} ms/step")
    
    total = t_ocean + t_atmos + t_land + t_ice
    print("-" * 30)
    print(f"Total Component Time: {total*1000:.2f} ms/step")
    print(f"Ocean Share: {t_ocean/total*100:.1f}%")
    print(f"Atmos Share: {t_atmos/total*100:.1f}%")

if __name__ == "__main__":
    profile_components()
