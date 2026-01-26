
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import data, main
from chronos_esm.atmos import dynamics as atmos_driver
from chronos_esm.atmos import physics as atmos_physics
from chronos_esm.coupler import state as coupled_state
from chronos_esm.ocean import veros_driver
from chronos_esm.config import CP_AIR

def setup_control_run():
    print("Loading initial conditions...")
    temp_ic, salt_ic = data.load_initial_conditions(nz=15)
    ny, nx = temp_ic.shape[1], temp_ic.shape[2]
    nz = temp_ic.shape[0]

    ocean = veros_driver.OceanState(
        u=jnp.zeros((nz, ny, nx)),
        v=jnp.zeros((nz, ny, nx)),
        w=jnp.zeros((nz, ny, nx)),
        temp=temp_ic + 273.15,
        salt=salt_ic,
        psi=jnp.zeros((ny, nx)),
        rho=jnp.zeros((nz, ny, nx)),
        dic=jnp.ones((nz, ny, nx)) * 2000.0,
    )
    atmos = atmos_driver.init_atmos_state(ny, nx)
    land = data.load_bathymetry_mask() # Dummy land (using mask as placeholder if needed, or proper init)
    from chronos_esm.land import driver as land_driver # re-import
    land = land_driver.init_land_state(ny, nx)
    
    state = coupled_state.init_coupled_state(ocean, atmos, land)
    # SST
    fluxes = state.fluxes._replace(sst=temp_ic[0] + 273.15)
    state = state._replace(fluxes=fluxes)
    return state

def diagnose_step():
    state = setup_control_run()
    regridder = main.regrid.Regridder()
    mask = data.load_bathymetry_mask()
    params = main.ModelParams(mask=mask)

    # We want to inspect the internal terms of step_atmos
    # We'll just run one coupled step and print the magnitudes of the terms in the PREVIOUS/Intermediate state
    # But it's easier to just copy the logic or inspect values if we break it down.
    
    # Or strict: Run 10 steps, check magnitudes.
    
    print(f"CP_AIR (from config): {CP_AIR}")
    
    # Let's inspect atmospheric physics directly
    print("\n--- Diagnostic: Atmos Physics Terms ---")
    
    # Mock data
    temp = 300.0 * jnp.ones((96, 192))
    co2 = 280.0
    
    # 1. Radiative Cooling
    cooling = atmos_physics.compute_radiative_forcing(temp, co2)
    mean_cooling = jnp.mean(jnp.abs(cooling))
    print(f"Mean Radiative Cooling (K/s): {mean_cooling:.2e}")
    
    # 2. Precip Heating
    q = 0.02 * jnp.ones((96, 192))
    q_sat = 0.01 * jnp.ones((96, 192)) # Supersaturated
    precip, heating = atmos_physics.compute_precipitation(q, q_sat)
    mean_heating = jnp.mean(heating)
    print(f"Mean Precip Heating (K/s): {mean_heating:.2e}")
    
    # 3. Sensible Heat Flux
    sensible, latent = atmos_physics.compute_surface_fluxes(
        temp_air=300.0, q_air=0.01, u_air=10.0, v_air=0.0,
        temp_surf=305.0 # Warmer surface
    )
    mean_sensible = jnp.mean(sensible) # W/m2
    print(f"Mean Sensible Flux (W/m2): {mean_sensible:.2e}")
    
    # 4. Dynamics Term Scaling
    # The code in dynamics.py is:
    # forcing_t = (heating_precip + heating_rad) + flux_sensible * mass_scaling / CP
    
    term_1_raw = mean_cooling + mean_heating # K/s (Already in K/s)
    
    # Sensible is W/m2. Need to convert to K/s.
    # forcing_sensible = flux_sensible * (1/Mass) / CP
    MASS_COLUMN = 1.0e4
    term_2_raw = mean_sensible # W/m2
    term_2_scaled = term_2_raw * (1.0 / MASS_COLUMN) / CP_AIR
    
    print("\n--- Dynamics.py Equation Check ---")
    print(f"Term 1 (Physics K/s): {term_1_raw:.2e}")
    print(f"Term 2 Scaled (Sensible -> K/s): {term_2_scaled:.2e}")
    
    ratio = term_2_scaled / (jnp.abs(term_1_raw) + 1e-20)
    print(f"Ratio (Sensible / Physics): {ratio:.2f}")
    
    if ratio > 100.0:
        print("\n[CRITICAL] Scaling Mismatch Detected!")
        print("Sensible heating dominates Physics by factor > 100 because of division by CP.")
        print("Radiative cooling is effectively disabled.")
    else:
        print("\nScaling seems comparable.")

if __name__ == "__main__":
    diagnose_step()
