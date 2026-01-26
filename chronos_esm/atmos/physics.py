"""
Differentiable physical parameterizations for the Atmosphere.

Includes:
1. Softplus-based precipitation (convection)
2. Radiative cooling
3. Anthropogenic forcing (CO2)
"""

from typing import Tuple

import jax
import jax.numpy as jnp

# Constants
QC_REF = 0.4915  # Tuned via JAX (Target: Idealized ITCZ)
TAU_CONV = 1800.0  # Fast convective adjustment (30 mins)
RAD_COOL_RATE = 1.0 / (86400.0 * 20.0)  # 20-day radiative damping

from chronos_esm.config import CP_AIR, DRAG_COEFF_OCEAN
EPSILON_SMOOTH = 4.71e-2 # Tuned to 0.0471
CO2_REF = 280.0  # Pre-industrial CO2 [ppm]
ALPHA_CO2 = 5.35  # Radiative forcing coefficient [W/m^2]


def compute_saturation_humidity(
    temp: jnp.ndarray, pressure: jnp.ndarray
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
    q: jnp.ndarray, q_sat: jnp.ndarray, epsilon: float = EPSILON_SMOOTH, 
    qc_ref: float = QC_REF
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute precipitation and heating using a differentiable softplus trigger.

    Standard: P = (q - q_c)/tau if q > q_c else 0
    Smooth:   P = softplus((q - q_c)/eps) * eps / tau

    Args:
        q: Specific humidity
        q_sat: Saturation specific humidity
        epsilon: Smoothing parameter
        qc_ref: Relative humidity threshold (0-1)

    Returns:
        precip: Precipitation rate [kg/kg/s]
        heating: Latent heating rate [K/s]
    """
    # Relative humidity-based threshold
    q_critical = qc_ref * q_sat

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
    sw_down: jnp.ndarray = None, 
    lat_rad: jnp.ndarray = None, 
    solar_constant: float = 1361.0
) -> jnp.ndarray:
    """
    Compute radiative heating rates including Anthropogenic Forcing.
    
    1. Newtonian cooling to equilibrium profile
    2. CO2 forcing: F = 5.35 * ln(C/C0)
    
    Args:
        temp: Temperature field [K]
        co2_ppm: Current CO2 concentration [ppm]
        sw_down: Downward Shortwave Flux [W/m^2] (Seasonal forcing)
        lat_rad: Latitude in radians (fallback if sw_down not provided)
        solar_constant: Solar constant (fallback scaling)
        
    Returns:
        heating_rate: Radiative heating rate [K/s]
    """
    # 1. Newtonian Cooling
    # Relax towards a simple radiative equilibrium T_eq
    
    # Standard GCM cooling: ~10-30 days. 
    # Tuned to 20 days for balance between gradient maintenance and ocean influence.
    rad_cool_rate = 1.0 / (86400.0 * 20.0)

    if sw_down is not None:
        # Drive T_eq with provided Shortwave Flux (Seasonal)
        # T_eq ~ 210K (Base/Night) + Sensitivity * SW_Down
        # Approx fit: 250 W/m2 (Surf) -> ~310K (Target).
        t_eq = 240.0 + 0.30 * sw_down
        
        # Ensure non-negative/stable
        t_eq = jnp.clip(t_eq, 180.0, 350.0)
        
    elif lat_rad is not None:
        # Fallback: Annual Mean Profile
        # Broadcast lat_rad (ny,) to (ny, nx)
        lat_2d = jnp.broadcast_to(lat_rad[:, None], temp.shape)
        
        # Reference Equilibrium at S0=1361.0
        # Base Equator Temp = 315.0K (Surface) to fix cold bias
        t_eq_ref = 315.0 - 60.0 * jnp.sin(lat_2d) ** 2
        
        # Scale with Solar Constant
        solar_scaling = (solar_constant / 1361.0) ** 0.25
        t_eq = t_eq_ref * solar_scaling
    else:
        # Fallback (dangerous for stability)
        t_eq = 288.0 * (solar_constant / 1361.0) ** 0.25

    # Newtonian cooling
    cooling = -rad_cool_rate * (temp - t_eq)

    # 2. Anthropogenic Forcing (Greenhouse Effect)
    # F_co2 [W/m^2]
    
    # Logarithmic forcing
    co2_safe = jnp.maximum(co2_ppm, 1.0)
    forcing_flux = ALPHA_CO2 * jnp.log(co2_safe / CO2_REF)

    # Convert to heating rate [K/s]
    # Distributed over column
    column_heat_capacity = 1.0e7
    forcing_rate = forcing_flux / column_heat_capacity

    return cooling + forcing_rate

def compute_solar_insolation(
    lat_rad: jnp.ndarray, # 1D array of latitudes
    day_of_year: float,
    solar_constant: float = 1361.0,
    eccentricity_factor: float = 1.0 # (r_mean / r)^2, approx 1.0 or use eccentricity
) -> jnp.ndarray:
    """
    Compute daily average Top-Of-Atmosphere solar insolation.
    
    Args:
        lat_rad: Latitude in radians
        day_of_year: Day of year (0-365)
        solar_constant: Solar constant [W/m2]
        
    Returns:
        insolation: Daily mean solar flux [W/m2] shape (nlat,)
    """
    # Solar Declination (delta)
    # Approx: delta = -23.44 * cos(360/365 * (N + 10))
    # Radians
    tilt = 23.44 * (jnp.pi / 180.0)
    # Phase shift: Winter solstice near day 355? 
    # Standard approx: delta = -23.44 * cos(2*pi * (day + 10)/365)
    # day 0 = Jan 1. Day 172 = June 21 (Summer Solstice).
    # cos((0+10)/365 * 2pi) ~ cos(0) ~ 1 -> Negative delta? 
    # Wait, -23.44 * 1 = -23.44 (Dec Solstice). Correct.
    
    delta = -tilt * jnp.cos(2 * jnp.pi * (day_of_year + 10.0) / 365.0)
    
    # Hour Angle at sunset (h0)
    # cos(h0) = -tan(phi)*tan(delta)
    # Clamp -1 to 1 for polar day/night
    tan_phi_delta = jnp.tan(lat_rad) * jnp.tan(delta)
    h0 = jnp.arccos((-tan_phi_delta).clip(-1.0, 1.0))
    
    # Daily Average Insolation
    # S = S0/pi * (h0*sin(phi)*sin(delta) + cos(phi)*cos(delta)*sin(h0))
    # S0 = Solar Constant
    
    term1 = h0 * jnp.sin(lat_rad) * jnp.sin(delta)
    term2 = jnp.cos(lat_rad) * jnp.cos(delta) * jnp.sin(h0)
    
    insolation = (solar_constant / jnp.pi) * (term1 + term2)
    
    # Apply Albedo Feedback
    albedo = compute_albedo(lat_rad)
    
    sw_surface = insolation * (1.0 - albedo) * 0.60
    
    return sw_surface


def compute_albedo(lat_rad: jnp.ndarray) -> jnp.ndarray:
    """
    Compute latitude-dependent albedo (planetary).
    
    Proxy for Cloud + Surface (Ice/Ocean) Albedo.
    
    Logic:
    - Tropics (Ocean dominated): Low Albedo (~0.05)
    - Poles (Ice/Snow/Cloud angle): High Albedo (~0.65)
    - Smooth sine-squared transition.
    
    Formula: A = 0.05 + 0.6 * sin(lat)^2
    Global Mean ~ 0.25.
    """
    return 0.05 + 0.6 * jnp.sin(lat_rad)**2


def compute_surface_fluxes(
    temp_air: jnp.ndarray,  # Lowest level air temp
    q_air: jnp.ndarray,  # Lowest level humidity
    u_air: jnp.ndarray,  # Lowest level wind speed (magnitude)
    v_air: jnp.ndarray,
    temp_surf: jnp.ndarray,  # Surface temperature
    beta: jnp.ndarray = 1.0,  # Evaporation efficiency (1 for ocean, <1 for land)
    rho_air: float = 1.2,
    cd: float = DRAG_COEFF_OCEAN,  # Drag coefficient
    ch: float = DRAG_COEFF_OCEAN,  # Heat exchange coefficient
    ce: float = DRAG_COEFF_OCEAN,  # Moisture exchange coefficient
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
    
    # Cap wind speed for fluxes to prevent runaway WISHE instability
    # (High winds -> High Evap -> High Latent Heat -> High Winds)
    wind_speed = jnp.minimum(wind_speed, 25.0)

    # Sensible Heat
    # SH = rho * Cp * Ch * U * (Ts - Ta)
    sensible_flux = rho_air * CP_AIR * ch * wind_speed * (temp_surf - temp_air)

    # Latent Heat
    # q_sat at surface temperature
    # Clausius-Clapeyron approx: q_sat = 0.622 * e_sat / p
    # e_sat = 611 * exp(17.27 * (T - 273.15) / (T - 35.85))
    t_c = temp_surf - 273.15
    e_sat = 611.0 * jnp.exp(17.27 * t_c / (t_c + 237.3))
    p_surf = 101325.0  # Approx surface pressure
    q_sat_surf = 0.622 * e_sat / p_surf

    # LH = rho * Lv * Ce * U * (qs - qa) * beta
    L_v = 2.5e6
    latent_flux = rho_air * L_v * ce * wind_speed * (q_sat_surf - q_air) * beta

    return sensible_flux, latent_flux
