"""
Driver for the Land Surface component.

Implements a simple Bucket Model for soil moisture and surface energy balance.
"""

from functools import partial
from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from chronos_esm.config import (
    ALBEDO_LAND,
    ATMOS_GRID,
    DRAG_COEFF_LAND,
    DT_ATMOS,
    EMISSIVITY_LAND,
    STEFAN_BOLTZMANN,
)
from chronos_esm.land import vegetation

# Constants
RHO_WATER = 1000.0
CP_SOIL = 800.0  # J/kg/K (approx for dry soil)
RHO_SOIL = 1600.0  # kg/m3
SOIL_DEPTH = 1.0  # meters (thermal depth) - Increased for stability
BUCKET_DEPTH = 0.15  # meters (water holding capacity)
L_VAPOR = 2.5e6



class LandState(NamedTuple):
    """State of the land surface."""

    temp: jnp.ndarray  # Surface temperature [K]
    soil_moisture: jnp.ndarray  # Soil moisture depth [m] (0 to BUCKET_DEPTH)
    lai: jnp.ndarray  # Leaf Area Index [m2/m2]
    soil_carbon: jnp.ndarray  # Soil Carbon [kg C/m2]
    snow_depth: jnp.ndarray  # Snow depth [m water equivalent]


def init_land_state(ny, nx) -> LandState:
    """Initialize land state."""
    return LandState(
        temp=jnp.ones((ny, nx)) * 290.0,
        soil_moisture=jnp.ones((ny, nx)) * (BUCKET_DEPTH * 0.5),  # 50% saturation
        lai=jnp.ones((ny, nx)) * 1.0,  # Initial sparse vegetation
        soil_carbon=jnp.ones((ny, nx)) * 10.0,  # 10 kg C/m2
        snow_depth=jnp.zeros((ny, nx)), # No initial snow
    )


@partial(jax.jit, static_argnames=["ny", "nx"])
def step_land(
    state: LandState,
    t_air: jnp.ndarray,  # Air temp [K]
    q_air: jnp.ndarray,  # Specific humidity [kg/kg]
    sw_down: jnp.ndarray,  # Shortwave down [W/m2]
    lw_down: jnp.ndarray,  # Longwave down [W/m2]
    precip: jnp.ndarray,  # Precipitation [m/s]
    mask: jnp.ndarray,  # Land mask (0=Ocean, 1=Land)
    
    # New Inputs
    wind_speed: jnp.ndarray, # [m/s]
    drag_coeff: jnp.ndarray, # [-]
    
    ny: int = ATMOS_GRID.nlat,
    nx: int = ATMOS_GRID.nlon,
) -> Tuple[LandState, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Time step the land model with Snow and Variable Roughness.
    """

    # 0. Update Vegetation
    lai_new = vegetation.step_vegetation(
        state.lai, state.temp, state.soil_moisture, BUCKET_DEPTH
    )

    # Get dynamic properties
    albedo_veg, veg_fraction = vegetation.compute_land_properties(lai_new)
    
    # Snow Mask
    # If snow depth > threshold (e.g. 1cm = 0.01m), albedo -> Snow Albedo
    SNOW_ALBEDO = 0.80
    snow_cover_fraction = jnp.clip(state.snow_depth / 0.05, 0.0, 1.0) # Fully covered at 5cm
    
    albedo = (1.0 - snow_cover_fraction) * albedo_veg + snow_cover_fraction * SNOW_ALBEDO

    # 1. Surface Energy Balance
    sw_net = sw_down * (1.0 - albedo)
    lw_up = EMISSIVITY_LAND * STEFAN_BOLTZMANN * state.temp**4
    lw_net = lw_down - lw_up
    r_net = sw_net + lw_net

    # Turbulent Fluxes using PROVIDED drag coefficient and wind speed
    # H = rho * Cp * Ch * U * (Ts - Ta)
    ch = drag_coeff
    ce = drag_coeff # Assuming scalar roughness for heat/moisture matches momentum for now
    
    rho_air = 1.225
    cp_air = 1004.0

    # Sensible Heat (Positive Upward)
    # Ensure wind is not zero to avoid singular drag (physically mixing is ch*U)
    # We assume 'drag_coeff' already encapsulates roughness logic
    sensible_flux = rho_air * cp_air * ch * wind_speed * (state.temp - t_air)

    # Latent Heat
    t_c = state.temp - 273.15
    es = 611.0 * jnp.exp(17.67 * t_c / (t_c + 243.5))
    p_surf = 101325.0
    q_sat = 0.622 * es / p_surf

    # Beta factor
    bucket_beta = state.soil_moisture / BUCKET_DEPTH
    bucket_beta = jnp.clip(bucket_beta, 0.0, 1.0)
    beta = bucket_beta * (1.0 + veg_fraction)
    beta = jnp.clip(beta, 0.0, 1.0)
    
    # If snow covered, allow evaporation/sublimation freely (beta=1)
    beta = (1.0 - snow_cover_fraction) * beta + snow_cover_fraction * 1.0

    evap_pot = rho_air * ce * wind_speed * (q_sat - q_air)
    evap_pot = jnp.maximum(evap_pot, 0.0)
    evap = beta * evap_pot
    
    # Latent Heat (Sublimation if T<0, Evap if T>0 - simplified L_VAPOR)
    latent_flux = L_VAPOR * evap

    # Net Flux into Soil
    g_flux = r_net - sensible_flux - latent_flux

    # 2. Snow vs Rain Logic
    # If T_air < 0C, precip is snow.
    is_snow = t_air < 273.15
    snow_fall = precip * is_snow
    rain_fall = precip * (1.0 - is_snow)
    
    # 3. Temperature Update (Semi-Implicit)
    # Similar to previous...
    d_rn_dt = -4 * EMISSIVITY_LAND * STEFAN_BOLTZMANN * state.temp**3
    d_h_dt = -rho_air * cp_air * ch * wind_speed
    d_qs_dt = q_sat * (17.67 * 243.5) / ((t_c + 243.5) ** 2)
    d_le_dt = -rho_air * L_VAPOR * ce * wind_speed * beta * d_qs_dt
    d_g_dt = d_rn_dt + d_h_dt + d_le_dt

    # Heat Capacity: Soil + Snow
    # Snow has low heat capacity effectively insulating soil?
    # Simplified: Add snow thermal inertia
    # C_snow approx 2000 J/kg/K * rho_snow (300) * depth
    heat_capacity_soil = RHO_SOIL * CP_SOIL * SOIL_DEPTH
    heat_capacity_snow = 300.0 * 2000.0 * state.snow_depth
    total_hc = heat_capacity_soil + heat_capacity_snow

    denominator = (total_hc / DT_ATMOS) - d_g_dt
    delta_temp = g_flux / denominator
    
    # Apply update on land
    temp_new = state.temp + delta_temp * mask
    
    # Check for Snow Melt
    # If Temp > 0C and Snow > 0
    # Melt Energy available = (Temp - 0) * HC
    # Actually, if T_new > 0 and we have snow, we peg T to 0 and use energy to melt.
    
    melt_energy = (temp_new - 273.15) * total_hc
    
    # Only if warm and snowy
    can_melt = (temp_new > 273.15) & (state.snow_depth > 0)
    
    # Amount melted [kg/m2] = Energy / L_fusion (3.34e5)
    L_FUSION = 3.34e5
    potential_melt_mass = melt_energy / L_FUSION
    potential_melt_mass = jnp.maximum(potential_melt_mass, 0.0) * can_melt
    
    # Converting to depth m (rho_water = 1000)
    potential_melt_depth = potential_melt_mass / RHO_WATER
    
    # Actual melt cannot exceed snow depth
    actual_melt_depth = jnp.minimum(potential_melt_depth, state.snow_depth)
    
    # Update Snow Depth
    snow_depth_new = state.snow_depth + DT_ATMOS * snow_fall - actual_melt_depth
    snow_depth_new = jnp.maximum(snow_depth_new, 0.0)
    
    # Update Temp: If melting occurred, T stays at 0C (273.15)
    # If we melted ALL snow, T can rise above 0C? 
    # Simplified: If potential > actual (all snow melted), T rises.
    # If potential < actual (snow remains), T = 0C.
    
    # T_corrected = 273.15 if snow remains, else calculated
    # Actually simpler: T_new -= (Melt_Energy_Used / HC)
    melt_energy_used = actual_melt_depth * RHO_WATER * L_FUSION
    temp_corrected = temp_new - (melt_energy_used / total_hc)
    
    # Only apply correction where relevant
    temp_new = jnp.where(can_melt, temp_corrected, temp_new)
    
    # Safety Clamping
    temp_new = jnp.clip(temp_new, 180.0, 340.0)
    temp_new = jnp.where(mask > 0.5, temp_new, state.temp)

    # 4. Soil Moisture Update
    # Inflow = Rain + SnowMelt
    evap_rate = evap / RHO_WATER
    water_in = rain_fall + (actual_melt_depth / DT_ATMOS) # melt is already per step amount, convert to rate? No wait.
    # actual_melt_depth is total meters in this step.
    # So rate = actual_melt_depth / DT
    
    dw_dt = (rain_fall + actual_melt_depth/DT_ATMOS) - evap_rate
    moisture_new = state.soil_moisture + DT_ATMOS * dw_dt * mask
    
    runoff = jnp.maximum(moisture_new - BUCKET_DEPTH, 0.0)
    moisture_new = jnp.minimum(moisture_new, BUCKET_DEPTH)
    moisture_new = jnp.maximum(moisture_new, 0.0)
    
    moisture_new = jnp.where(mask > 0.5, moisture_new, state.soil_moisture)
    lai_new = jnp.where(mask > 0.5, lai_new, state.lai)
    snow_depth_new = jnp.where(mask > 0.5, snow_depth_new, state.snow_depth)

    # 5. Carbon Cycle (Simplified)
    lue = 1.0e-9
    gpp = lue * sw_down * lai_new * beta
    r_auto = 0.5 * gpp
    t_ref = 288.15
    q10_factor = 2.0 ** ((state.temp - t_ref) / 10.0)
    k_soil = 1.0e-8
    r_hetero = k_soil * state.soil_carbon * q10_factor * beta
    nee = r_auto + r_hetero - gpp
    litter = gpp - r_auto
    d_soil_c_dt = litter - r_hetero
    soil_carbon_new = state.soil_carbon + DT_ATMOS * d_soil_c_dt * mask
    soil_carbon_new = jnp.maximum(soil_carbon_new, 0.0)
    soil_carbon_new = jnp.where(mask > 0.5, soil_carbon_new, state.soil_carbon)

    new_state = LandState(
        temp=temp_new,
        soil_moisture=moisture_new,
        lai=lai_new,
        soil_carbon=soil_carbon_new,
        snow_depth=snow_depth_new,
    )

    return new_state, (sensible_flux, latent_flux, nee)
