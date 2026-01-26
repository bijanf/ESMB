
import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
import numpy as np
from chronos_esm import main, data, config
from chronos_esm.ocean import veros_driver
from chronos_esm.atmos import dynamics as atmos_driver
from chronos_esm.land import driver as land_driver
from chronos_esm.coupler import state as coupled_state

def setup_state():
    print("Loading ICs...")
    temp_ic, salt_ic = data.load_initial_conditions(nz=config.OCEAN_GRID.nz)
    
    # Initialize components
    ocean = veros_driver.init_ocean_state(config.OCEAN_GRID.nz, config.OCEAN_GRID.nlat, config.OCEAN_GRID.nlon)
    ocean = ocean._replace(temp=temp_ic + 273.15, salt=salt_ic) # Load WOA
    
    atmos = atmos_driver.init_atmos_state(config.ATMOS_GRID.nlat, config.ATMOS_GRID.nlon)
    land = land_driver.init_land_state(config.ATMOS_GRID.nlat, config.ATMOS_GRID.nlon)
    
    state = coupled_state.init_coupled_state(ocean, atmos, land)
    # Sync SST
    state = state._replace(fluxes=state.fluxes._replace(sst=state.ocean.temp[0] - 273.15))
    
    return state

def debug_run():
    state = setup_state()
    params = main.ModelParams(mask=data.load_bathymetry_mask(), co2_increase_rate=0.0)
    regridder = main.regrid.Regridder()
    
    print("state.ocean.temp min/max:", state.ocean.temp.min(), state.ocean.temp.max())
    
    mask_min = float(params.mask.min())
    mask_max = float(params.mask.max())
    print(f"Mask: [{mask_min}, {mask_max}]")
    
    soil_min = float(state.land.soil_moisture.min())
    soil_max = float(state.land.soil_moisture.max())
    print(f"Soil Moisture: [{soil_min}, {soil_max}]")
    
    # Run 500 coupled steps
    for i in range(1, 501):
        print(f"\n--- Coupled Step {i} ---")
        # We call step_coupled (which does 30 atmos steps internally)
        # It is already jitted in main.py
        state = main.step_coupled(state, params, regridder)
        
        # Block and check
        state.atmos.temp.block_until_ready()
        
        t_min = float(state.atmos.temp.min())
        t_max = float(state.atmos.temp.max())
        o_min = float(state.ocean.temp.min())
        o_max = float(state.ocean.temp.max())
        
        flux_heat_min = float(state.fluxes.net_heat_flux.min())
        flux_heat_max = float(state.fluxes.net_heat_flux.max())
        tau_x_max = float(jnp.abs(state.fluxes.wind_stress_x).max())
        
        u_max = float(jnp.abs(state.atmos.u).max())
        
        ln_ps_min = float(state.atmos.ln_ps.min())
        ln_ps_max = float(state.atmos.ln_ps.max())
        
        q_min = float(state.atmos.q.min())
        q_max = float(state.atmos.q.max())
        
        sst_min = float(state.fluxes.sst.min())
        sst_max = float(state.fluxes.sst.max())
        
        print(f"Atmos Temp: [{t_min:.2f}, {t_max:.2f}]")
        print(f"Atmos ln_ps: [{ln_ps_min:.2f}, {ln_ps_max:.2f}]")
        print(f"Atmos q: [{q_min:.2e}, {q_max:.2e}]")
        print(f"SST (Composite): [{sst_min:.2f}, {sst_max:.2f}]")
        print(f"Ocean Temp: [{o_min:.2f}, {o_max:.2f}]")
        print(f"Heat Flux (Down): [{flux_heat_min:.2f}, {flux_heat_max:.2f}]")
        print(f"Wind Stress X Max: {tau_x_max:.4f}")
        print(f"Atmos U Max: {u_max:.2f}")
        
        # Decompose Flux loop
        from chronos_esm.atmos import physics as atoms_phys
        from chronos_esm.land import vegetation
        
        # Calculate Beta
        ocean_mask = params.mask
        land_mask = 1.0 - ocean_mask
        BUCKET_DEPTH = 0.15
        _, veg_fraction = vegetation.compute_land_properties(state.land.lai)
        bucket_beta = state.land.soil_moisture / BUCKET_DEPTH
        bucket_beta = jnp.clip(bucket_beta, 0.0, 1.0)
        land_beta = bucket_beta * (1.0 + veg_fraction)
        land_beta = jnp.clip(land_beta, 0.0, 1.0)
        beta = ocean_mask * 1.0 + land_mask * land_beta
        
        beta_min = float(beta.min())
        beta_max = float(beta.max())
        print(f"  Beta: [{beta_min}, {beta_max}]")

        sens, lat = atoms_phys.compute_surface_fluxes(
             state.atmos.temp, state.atmos.q, state.atmos.u, state.atmos.v,
             state.fluxes.sst,
             beta=beta
        ) 
        
        sens_min = float(sens.min())
        sens_max = float(sens.max())
        lat_min = float(lat.min())
        lat_max = float(lat.max())
        
        print(f"  Manual Sensible: [{sens_min:.2f}, {sens_max:.2f}]")
        print(f"  Manual Latent:   [{lat_min:.2f}, {lat_max:.2f}]")
        
        if np.isnan(u_max) or np.isnan(tau_x_max):
             print("WIND/STRESS NaN DETECTED!")
             
        if np.isnan(t_min) or np.isnan(o_min):
            print("NaN DETECTED!")
            # Break down which component
            if np.isnan(t_min): print("Atmos is NaN")
            if np.isnan(o_min): print("Ocean is NaN")
            break

if __name__ == "__main__":
    debug_run()
