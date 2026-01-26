
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
import time
from chronos_esm import main, data
from chronos_esm.ocean import diagnostics

def run_realistic():
    print("Initializing Realistic Simulation...")
    
    # 1. Setup Parameters
    mask = jnp.array(data.load_bathymetry_mask())
    params = main.ModelParams(mask=mask, co2_ppm=280.0)
    
    # 2. Initialize State
    state = main.init_model()
    regridder = main.regrid.Regridder()
    
    # 3. Parameters
    r_drag = 1.0e-2 # Increased for stability (was 2.5e-5)
    print(f"Running with r_drag = {r_drag}")
    print(f"Initial AMOC: {diagnostics.compute_amoc_index(state.ocean):.2f} Sv")
    
    # 4. Run Loop
    steps_per_year = int(365 * 24 * 3600 / 900) # ~35040
    print(f"Steps per year: {steps_per_year}")
    
    print("Starting integration...")
    t0 = time.time()
    
    # Correct step function handling tuple return
    def step_fn(carry, _):
        s = carry
        # step_coupled returns CoupledState
        new_state = main.step_coupled(s, params, regridder, r_drag=r_drag)
        return new_state, None
        
    final_state, _ = jax.lax.scan(step_fn, state, None, length=steps_per_year)
    final_state.ocean.temp.block_until_ready()
    t1 = time.time()
    
    print(f"Integration complete in {t1-t0:.2f} s")
    print(f"Speed: {steps_per_year / (t1-t0):.2f} steps/s")
    
    # 5. Analyze Results
    
    # AMOC
    amoc = diagnostics.compute_amoc_index(final_state.ocean)
    print(f"Final AMOC: {amoc:.2f} Sv")
    
    # SST (Ocean Surface)
    sst_mean = jnp.nanmean(jnp.where(mask, final_state.fluxes.sst, jnp.nan)) - 273.15
    print(f"Global Mean SST: {sst_mean:.2f} C")
    
    # SSS
    sss_mean = jnp.mean(final_state.ocean.salt[0])
    print(f"Global Mean SSS: {sss_mean:.2f} psu")
    
    # SAT (Land Temperature)
    land_mask = 1.0 - mask
    sat_mean = jnp.nanmean(jnp.where(land_mask, final_state.land.temp, jnp.nan)) - 273.15
    print(f"Global Mean SAT (Land): {sat_mean:.2f} C")
    
    # Precipitation / Monsoon (P-E)
    # fw_flux = Precip - Evap. (Positive into ocean/land).
    fw_mean_ocean = jnp.nanmean(jnp.where(mask, final_state.fluxes.freshwater_flux, jnp.nan))
    fw_mean_land = jnp.nanmean(jnp.where(land_mask, final_state.fluxes.freshwater_flux, jnp.nan))
    
    print(f"Global Mean P-E (Ocean): {fw_mean_ocean:.2e} m/s")
    print(f"Global Mean P-E (Land): {fw_mean_land:.2e} m/s")
    
    # Check for "High" Precip over land (Monsoon signal?)
    # Just max value
    max_pe_land = jnp.max(jnp.where(land_mask, final_state.fluxes.freshwater_flux, -1e9))
    print(f"Max P-E over Land: {max_pe_land:.2e} m/s")

if __name__ == "__main__":
    run_realistic()
