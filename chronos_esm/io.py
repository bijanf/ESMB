"""
Input/Output module for Chronos-ESM.

Handles saving model state to NetCDF.
"""

from pathlib import Path
from typing import Union

import jax.numpy as jnp
import numpy as np
import xarray as xr

from chronos_esm.config import ATMOS_GRID, OCEAN_GRID
from chronos_esm.coupler import state as coupled_state


def save_state_to_netcdf(state: coupled_state.CoupledState, filepath: Union[str, Path]):
    """
    Save the coupled state to a NetCDF file.

    Args:
        state: The coupled model state.
        filepath: Path to save the file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert JAX arrays to numpy
    # Ocean
    temp_ocn = np.array(state.ocean.temp)
    salt_ocn = np.array(state.ocean.salt)
    u_ocn = np.array(state.ocean.u)
    v_ocn = np.array(state.ocean.v)
    w_ocn = np.array(state.ocean.w)

    # Ice
    h_ice = np.array(state.ice.thickness)
    a_ice = np.array(state.ice.concentration)

    # Atmos
    temp_atm = np.array(state.atmos.temp)
    q_atm = np.array(state.atmos.q)
    u_atm = np.array(state.atmos.u)
    v_atm = np.array(state.atmos.v)

    # Coordinates
    nz, ny_ocn, nx_ocn = temp_ocn.shape
    ny_atm, nx_atm = temp_atm.shape

    # Create Dataset
    ds = xr.Dataset(
        data_vars={
            # Ocean
            "ocean_temp": (
                ("z", "y_ocn", "x_ocn"),
                temp_ocn,
                {"units": "K", "long_name": "Ocean Potential Temperature"},
            ),
            "ocean_salt": (
                ("z", "y_ocn", "x_ocn"),
                salt_ocn,
                {"units": "psu", "long_name": "Ocean Salinity"},
            ),
            "ocean_u": (
                ("z", "y_ocn", "x_ocn"),
                u_ocn,
                {"units": "m/s", "long_name": "Ocean Zonal Velocity"},
            ),
            "ocean_v": (
                ("z", "y_ocn", "x_ocn"),
                v_ocn,
                {"units": "m/s", "long_name": "Ocean Meridional Velocity"},
            ),
            # Ice
            "ice_thickness": (
                ("y_ocn", "x_ocn"),
                h_ice,
                {"units": "m", "long_name": "Sea Ice Thickness"},
            ),
            "ice_concentration": (
                ("y_ocn", "x_ocn"),
                a_ice,
                {"units": "1", "long_name": "Sea Ice Concentration"},
            ),
            # Atmos
            "atmos_temp": (
                ("y_atm", "x_atm"),
                temp_atm,
                {"units": "K", "long_name": "Atmospheric Temperature"},
            ),
            "atmos_q": (
                ("y_atm", "x_atm"),
                q_atm,
                {"units": "kg/kg", "long_name": "Specific Humidity"},
            ),
            "atmos_u": (
                ("y_atm", "x_atm"),
                u_atm,
                {"units": "m/s", "long_name": "Atmospheric Zonal Wind"},
            ),
            "atmos_v": (
                ("y_atm", "x_atm"),
                v_atm,
                {"units": "m/s", "long_name": "Atmospheric Meridional Wind"},
            ),
            # Saved Prognostic Variables (Critical for Restart)
            "atmos_ln_ps": (
                ("y_atm", "x_atm"),
                np.array(state.atmos.ln_ps),
                {"units": "log(Pa)", "long_name": "Log Surface Pressure"},
            ),
            "atmos_vorticity": (
                ("y_atm", "x_atm"),
                np.array(state.atmos.vorticity),
                {"units": "1/s", "long_name": "Relative Vorticity"},
            ),
            "atmos_divergence": (
                ("y_atm", "x_atm"),
                np.array(state.atmos.divergence),
                {"units": "1/s", "long_name": "Divergence"},
            ),
            "atmos_co2": (
                 ("y_atm", "x_atm"),
                 np.array(state.atmos.co2),
                 {"units": "ppm", "long_name": "CO2 Concentration"},
            ),
            # Land
            "land_temp": (
                ("y_atm", "x_atm"),
                np.array(state.land.temp),
                {"units": "K", "long_name": "Land Surface Temperature"},
            ),
            "land_snow_depth": (
                ("y_atm", "x_atm"),
                np.array(state.land.snow_depth),
                {"units": "m", "long_name": "Snow Depth (Water Equivalent)"},
            ),
            # Fluxes
            "sst": (
                 ("y_atm", "x_atm"),
                 np.array(state.fluxes.sst),
                 {"units": "C", "long_name": "Composite Surface Temperature"},
            ),
            "wind_stress_x": (
                 ("y_atm", "x_atm"),
                 np.array(state.fluxes.wind_stress_x),
                 {"units": "N/m^2", "long_name": "Zonal Wind Stress"},
            ),
            "wind_stress_y": (
                 ("y_atm", "x_atm"),
                 np.array(state.fluxes.wind_stress_y),
                 {"units": "N/m^2", "long_name": "Meridional Wind Stress"},
            ),
            "precip": (
                 ("y_atm", "x_atm"),
                 np.array(state.fluxes.precip),
                 {"units": "kg/m^2/s", "long_name": "Precipitation Rate"},
            ),
            # Time
            "time": ((), float(state.time), {"units": "seconds since start"}),
        },
        coords={
            "z": np.arange(nz),
            "y_ocn": np.arange(ny_ocn),
            "x_ocn": np.arange(nx_ocn),
            "y_atm": np.arange(ny_atm),
            "x_atm": np.arange(nx_atm),
        },
        attrs={
            "description": "Chronos-ESM Coupled State",
            "model_time": float(state.time),
        },
    )

    ds.to_netcdf(filepath)
    print(f"Saved state to {filepath}")


def load_state_from_netcdf(filepath: Union[str, Path]) -> coupled_state.CoupledState:
    """
    Load the coupled state from a NetCDF file.
    
    Args:
        filepath: Path to the NetCDF file.
        
    Returns:
        state: The reconstructed CoupledState.
    """
    ds = xr.open_dataset(filepath, decode_times=False)
    
    # Dimensions
    nz = ds.sizes['z']
    ny_ocn = ds.sizes['y_ocn']
    nx_ocn = ds.sizes['x_ocn']
    ny_atm = ds.sizes['y_atm']
    nx_atm = ds.sizes['x_atm']
    
    # --- Ocean ---
    # Load available prognostic variables
    temp = jnp.array(ds.ocean_temp.values)
    salt = jnp.array(ds.ocean_salt.values)
    u = jnp.array(ds.ocean_u.values)
    v = jnp.array(ds.ocean_v.values)
    
    # Missing diagnostics (re-init to zero/default)
    w = jnp.zeros((nz, ny_ocn, nx_ocn))
    psi = jnp.zeros((ny_ocn, nx_ocn))
    rho = jnp.zeros((nz, ny_ocn, nx_ocn))
    dic = jnp.ones((nz, ny_ocn, nx_ocn)) * 2000.0 # Default DIC
    
    ocean = coupled_state.OceanState(
        u=u, v=v, w=w, temp=temp, salt=salt, psi=psi, rho=rho, dic=dic
    )
    
    # --- Ice ---
    h_ice = jnp.array(ds.ice_thickness.values)
    a_ice = jnp.array(ds.ice_concentration.values)
    # Surface temp not saved in previous versions, default to -1.8 C
    if 'ice_surface_temp' in ds:
        surface_temp = jnp.array(ds.ice_surface_temp.values)
    else:
        surface_temp = jnp.ones_like(h_ice) * -1.8
    
    ice = coupled_state.IceState(
        thickness=h_ice,
        concentration=a_ice,
        surface_temp=surface_temp
    )
    
    # --- Atmos ---
    temp_atm = jnp.array(ds.atmos_temp.values)
    
    # q_atm loaded later after potential reset logic
    u_atm = jnp.array(ds.atmos_u.values)
    v_atm = jnp.array(ds.atmos_v.values)
    
    # Re-init diagnostics
    # Re-init diagnostics or load prognostics
    reset_humidity = False
    
    if 'atmos_ln_ps' in ds:
        ln_ps = jnp.array(ds.atmos_ln_ps.values)
    else:
        print("Warning: atmos_ln_ps not in restart. Reconstructing hydrostatically.")
        # Fallback to 101325 constant to avoid vacuum.
        ln_ps = jnp.ones((ny_atm, nx_atm)) * jnp.log(101325.0)
        reset_humidity = True # Vacuum state Q is likely garbage/too low for new pressure.

    # ... (Vorticity block same) ...

    # Humidity: if Reset needed (Rescue), re-initialize Q to 80% RH
    if reset_humidity:
        print("Warning: Reseting Humidity to 80% RH to recover from vacuum state.")
        from chronos_esm.atmos import physics
        pressure = jnp.exp(ln_ps)
        q_sat = physics.compute_saturation_humidity(temp_atm, pressure)
        q_atm = 0.8 * q_sat
    else:
        q_atm = jnp.array(ds.atmos_q.values)

    if 'atmos_vorticity' in ds and 'atmos_divergence' in ds:
        vorticity = jnp.array(ds.atmos_vorticity.values)
        divergence = jnp.array(ds.atmos_divergence.values)
    else:
        print("Warning: Prognostic Vort/Div not in restart. Reconstructing from U, V.")
        # Recompute from U, V
        from chronos_esm.atmos import spectral
        from chronos_esm.config import EARTH_RADIUS
        
        lat = jnp.linspace(-90, 90, ny_atm)
        lat_rad = jnp.deg2rad(lat)
        dx = 2 * jnp.pi * EARTH_RADIUS * jnp.cos(lat_rad)[:, None] / nx_atm
        dy = jnp.pi * EARTH_RADIUS / ny_atm
        
        du_dx, du_dy = spectral.compute_gradients(u_atm, dx, dy)
        dv_dx, dv_dy = spectral.compute_gradients(v_atm, dx, dy)
        
        vorticity = dv_dx - du_dy
        divergence = du_dx + dv_dy

    if 'atmos_co2' in ds:
        co2 = jnp.array(ds.atmos_co2.values)
    else:
        co2 = jnp.ones((ny_atm, nx_atm)) * 280e-6 # Pre-industrial
    
    # Initialize missing diagnostics
    psi_atm = jnp.zeros((ny_atm, nx_atm))
    chi_atm = jnp.zeros((ny_atm, nx_atm))
    phi_s_atm = jnp.zeros((ny_atm, nx_atm)) # Topography (should be static)

    atmos = coupled_state.AtmosState(
        u=u_atm, v=v_atm, temp=temp_atm, q=q_atm, 
        ln_ps=ln_ps, vorticity=vorticity, divergence=divergence, co2=co2,
        psi=psi_atm, chi=chi_atm, phi_s=phi_s_atm
    )

    # --- Land ---
    land_temp = jnp.array(ds.land_temp.values)
    land_snow = jnp.array(ds.land_snow_depth.values)
    
    # Defaults for missing fields
    BUCKET_DEPTH = 0.15
    land_sm = jnp.ones_like(land_temp) * (BUCKET_DEPTH * 0.5)
    land_lai = jnp.ones_like(land_temp) * 1.0
    land_carbon = jnp.ones_like(land_temp) * 10.0

    land = coupled_state.LandState(
        temp=land_temp,
        soil_moisture=land_sm,
        lai=land_lai,
        soil_carbon=land_carbon,
        snow_depth=land_snow
    )
    
    # --- Fluxes ---
    sst = jnp.array(ds.sst.values)
    
    if 'wind_stress_x' in ds:
        tau_x = jnp.array(ds.wind_stress_x.values)
        tau_y = jnp.array(ds.wind_stress_y.values)
        precip = jnp.array(ds.precip.values)
    else:
        tau_x = jnp.zeros((ny_atm, nx_atm))
        tau_y = jnp.zeros((ny_atm, nx_atm))
        precip = jnp.zeros((ny_atm, nx_atm))

    fluxes = coupled_state.FluxState(
        net_heat_flux=jnp.zeros((ny_atm, nx_atm)),
        freshwater_flux=jnp.zeros((ny_atm, nx_atm)),
        wind_stress_x=tau_x,
        wind_stress_y=tau_y,
        precip=precip,
        sst=sst,
        carbon_flux_ocean=jnp.zeros((ny_atm, nx_atm)),
        carbon_flux_land=jnp.zeros((ny_atm, nx_atm))
    )
    
    # --- Time ---
    # "seconds since start"
    time_val = float(ds.time.values)
    
    state = coupled_state.CoupledState(
        ocean=ocean,
        atmos=atmos,
        ice=ice,
        land=land,
        fluxes=fluxes,
        time=time_val
    )
    
    return state
