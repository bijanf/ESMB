
import jax
import jax.numpy as jnp
from chronos_esm import main, data
from chronos_esm.atmos import dynamics as atmos_driver
from chronos_esm.atmos import physics
from chronos_esm.config import ATMOS_GRID, DT_ATMOS
import numpy as np

def debug_step1():
    print("Initializing Model...")
    state = main.init_model()
    
    # Extract Atmos State
    atmos = state.atmos
    
    # Mock Surface Inputs (from Init)
    # Ocean 10C, Land ??? 
    # Use state coupling values
    sst = state.fluxes.sst # Combined surface temp
    
    # Calculate Fluxes manually (as in step_coupled)
    # We need to mimic step_coupled's preparation for step_atmos
    
    # Assume ocean mask
    mask = data.load_bathymetry_mask()
    
    # Define Temp Cold for Mock
    lat = jnp.linspace(-90, 90, 48)
    lat_rad = jnp.deg2rad(lat)
    temp_cold = 260.0 - 20.0 * jnp.sin(lat_rad)[:, None] ** 2
    temp_cold = jnp.broadcast_to(temp_cold, (48, 96))

    # 1. Fluxes
    # We need to call physics.compute_surface_fluxes
    
    # Wind speed
    print(f"U Mean: {jnp.mean(atmos.u):.2f} m/s")
    print(f"U Max: {jnp.max(atmos.u):.2f} m/s")
    
    wind_speed = jnp.sqrt(atmos.u**2 + atmos.v**2) + 1.0
    
    # Drag
    topo_height = atmos.phi_s / 9.81
    z0 = 0.0001 + 0.001 * topo_height
    z0 = jnp.clip(z0, 0.0001, 5.0)
    k = 0.4
    z_lev = 50.0
    cd = (k / jnp.log(z_lev / z0)) ** 2
    cd = jnp.maximum(cd, 1.0e-3)
    
    # Beta
    beta = jnp.ones_like(sst) # simplified
    
    sens, lat_flux = physics.compute_surface_fluxes(
        temp_air=temp_cold,
        q_air=atmos.q,
        u_air=atmos.u,
        v_air=atmos.v,
        temp_surf=sst,
        beta=beta,
        cd=cd, ch=cd, ce=cd
    )
    
    print(f"Sensible Flux Mean: {jnp.mean(sens):.2f} W/m2")
    # 2. Physics Tendencies

    t_rad = physics.compute_radiative_forcing(
        temp_cold, 
        co2_ppm=280.0, 
        lat_rad=lat_rad, 
        solar_constant=1361.0
    )
    
    print(f"Rad Heating Mean: {jnp.mean(t_rad):.2e} K/s")
    print(f"Rad Heating Min: {jnp.min(t_rad):.2e} K/s") # Should be negative (Cooling)
    
    # Precip
    pressure = jnp.exp(atmos.ln_ps)
    q_sat = physics.compute_saturation_humidity(atmos.temp, pressure)
    precip, t_precip = physics.compute_precipitation(atmos.q, q_sat)
    
    print(f"Precip Heating Mean: {jnp.mean(t_precip):.2e} K/s")
    print(f"Precip Heating Max: {jnp.max(t_precip):.2e} K/s")
    
    # Total
    total_tendency = t_rad + t_precip + sens / 1.0e4 / 1004.0
    
    print(f"Total T Tendency Mean: {jnp.mean(total_tendency):.2e} K/s")
    print(f"Total T Tendency Max: {jnp.max(total_tendency):.2e} K/s")

if __name__ == "__main__":
    debug_step1()
