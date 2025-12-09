"""
Differentiable physical parameterizations for the Atmosphere.

Includes:
1. Softplus-based precipitation (convection)
2. Radiative cooling
3. Anthropogenic forcing (CO2)
"""

import jax
import jax.numpy as jnp
from typing import Tuple

from chronos_esm.config import CP_AIR, EPSILON_SMOOTH, DRAG_COEFF_OCEAN

# Constants
QC_REF = 0.8         # Reference humidity threshold (normalized)
TAU_CONV = 3600.0    # Convective time scale [s]
RAD_COOL_RATE = 1.0 / (86400.0 * 10.0) # 10-day radiative damping
CO2_REF = 280.0      # Pre-industrial CO2 [ppm]
ALPHA_CO2 = 5.35     # Radiative forcing coefficient [W/m^2]


def compute_saturation_humidity(
    temp: jnp.ndarray,
    pressure: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute saturation specific humidity using Clausius-Clapeyron.
    """
    # e_sat = 611 * exp(17.27 * (T - 273.15) / (T - 35.85))
    t_c = temp - 273.15
    e_sat = 611.0 * jnp.exp(17.27 * t_c / (t_c + 237.3))
    
    # q_sat = 0.622 * e_sat / p
    q_sat = 0.622 * e_sat / pressure
    return q_sat


def compute_precipitation(
    q: jnp.ndarray,
    q_sat: jnp.ndarray,
    epsilon: float = EPSILON_SMOOTH
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute precipitation and heating using a differentiable softplus trigger.
    
    Standard: P = (q - q_c)/tau if q > q_c else 0
    Smooth:   P = softplus((q - q_c)/eps) * eps / tau
    
    Args:
        q: Specific humidity
        q_sat: Saturation specific humidity
        epsilon: Smoothing parameter
        
    Returns:
        precip: Precipitation rate [kg/kg/s]
        heating: Latent heating rate [K/s]
    """
    # Relative humidity-based threshold
    q_critical = QC_REF * q_sat
    
    # Excess humidity (normalized by epsilon for softplus)
    excess = (q - q_critical) / epsilon
    
    # Softplus: log(1 + exp(x))
    # For large x, approaches x. For large -x, approaches 0.
    # We multiply by epsilon/tau to get rate
    
    # Use Relu to avoid bias at 0
    # precip = jax.nn.relu(excess) * (epsilon / TAU_CONV)
    
    # Actually, let's keep softplus but shift it? 
    # Or just use relu. Relu is fine.
    precip = jax.nn.relu(excess) * (epsilon / TAU_CONV)
    
    # Latent heating: L_v * P / C_p
    # L_v approx 2.5e6 J/kg
    L_v = 2.5e6
    heating = (L_v / CP_AIR) * precip
    
    return precip, heating


def compute_radiative_forcing(
    temp: jnp.ndarray,
    co2_ppm: float,
    lat_rad: jnp.ndarray = None
) -> jnp.ndarray:
    """
    Compute radiative heating rates including Anthropogenic Forcing.
    
    1. Newtonian cooling to equilibrium profile
    2. CO2 forcing: F = 5.35 * ln(C/C0)
    
    Args:
        temp: Temperature field [K]
        co2_ppm: Current CO2 concentration [ppm]
        lat_rad: Latitude in radians (optional, but recommended for stability)
        
    Returns:
        heating_rate: Radiative heating rate [K/s]
    """
    # 1. Newtonian Cooling
    # Relax towards a simple radiative equilibrium T_eq
    # T_eq varies with latitude to drive circulation
    # T_eq = 315 - 60 * sin^2(lat) (approx)
    
    # Stronger cooling: 3 day damping
    rad_cool_rate = 1.0 / (86400.0 * 3.0)
    
    if lat_rad is not None:
        # Proper restoration profile
        # Broadcast lat_rad (ny,) to (ny, nx)
        lat_2d = jnp.broadcast_to(lat_rad[:, None], temp.shape)
        t_eq = 315.0 - 60.0 * jnp.sin(lat_2d)**2
    else:
        # Fallback (dangerous for stability)
        t_eq = 288.0 
    
    # Newtonian cooling
    cooling = -rad_cool_rate * (temp - t_eq)
    
    # 2. Anthropogenic Forcing (Greenhouse Effect)
    # F_co2 [W/m^2] = alpha * ln(CO2 / CO2_ref)
    # Heating rate [K/s] = F_co2 / (rho * Cp * H_scale)
    # We need to convert W/m^2 to K/s.
    # Assume heating is distributed over the column or lower atmosphere.
    # Column heat capacity C_col ~ 10^7 J/K/m^2?
    # Air mass ~ 10^4 kg/m^2. Cp ~ 1000. C_col ~ 10^7.
    # So 1 W/m^2 ~ 1e-7 K/s ~ 0.01 K/day.
    
    # Logarithmic forcing
    # Avoid log(0) or negative
    co2_safe = jnp.maximum(co2_ppm, 1.0)
    forcing_flux = ALPHA_CO2 * jnp.log(co2_safe / CO2_REF)
    
    # Convert to heating rate [K/s]
    # Distributed over column approx 10000 kg/m^2 * 1004 J/kg/K
    column_heat_capacity = 1.0e7 
    forcing_rate = forcing_flux / column_heat_capacity
    
    return cooling + forcing_rate


def compute_surface_fluxes(
    temp_air: jnp.ndarray, # Lowest level air temp
    q_air: jnp.ndarray,    # Lowest level humidity
    u_air: jnp.ndarray,    # Lowest level wind speed (magnitude)
    v_air: jnp.ndarray,
    temp_surf: jnp.ndarray, # Surface temperature
    beta: jnp.ndarray = 1.0, # Evaporation efficiency (1 for ocean, <1 for land)
    rho_air: float = 1.2,
    cd: float = DRAG_COEFF_OCEAN,     # Drag coefficient
    ch: float = DRAG_COEFF_OCEAN,     # Heat exchange coefficient
    ce: float = DRAG_COEFF_OCEAN      # Moisture exchange coefficient
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute surface sensible and latent heat fluxes using bulk formulas.
    
    Sensible Heat = rho * Cp * Ch * U * (Ts - Ta)
    Latent Heat   = rho * Lv * Ce * U * (qs - qa) * beta
    
    Args:
        temp_air: Air temperature [K]
        q_air: Specific humidity [kg/kg]
        u_air, v_air: Wind components [m/s]
        temp_surf: Surface temperature [K]
        beta: Evaporation efficiency (0-1)
        
    Returns:
        sensible_flux: [W/m^2] (Positive Upward)
        latent_flux:   [W/m^2] (Positive Upward)
    """
    # Wind speed magnitude
    wind_speed = jnp.sqrt(u_air**2 + v_air**2)
    # Minimum wind speed for fluxes (gustiness)
    wind_speed = jnp.maximum(wind_speed, 1.0)
    
    # Sensible Heat
    # SH = rho * Cp * Ch * U * (Ts - Ta)
    sensible_flux = rho_air * CP_AIR * ch * wind_speed * (temp_surf - temp_air)
    
    # Latent Heat
    # q_sat at surface temperature
    # Clausius-Clapeyron approx: q_sat = 0.622 * e_sat / p
    # e_sat = 611 * exp(17.27 * (T - 273.15) / (T - 35.85))
    t_c = temp_surf - 273.15
    e_sat = 611.0 * jnp.exp(17.27 * t_c / (t_c + 237.3))
    p_surf = 101325.0 # Approx surface pressure
    q_sat_surf = 0.622 * e_sat / p_surf
    
    # LH = rho * Lv * Ce * U * (qs - qa) * beta
    L_v = 2.5e6
    latent_flux = rho_air * L_v * ce * wind_speed * (q_sat_surf - q_air) * beta
    
    return sensible_flux, latent_flux
