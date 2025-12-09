"""
Driver for the Land Surface component.

Implements a simple Bucket Model for soil moisture and surface energy balance.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from functools import partial

from chronos_esm.config import ATMOS_GRID, DT_ATMOS, ALBEDO_LAND, STEFAN_BOLTZMANN, DRAG_COEFF_LAND, EMISSIVITY_LAND
from chronos_esm.land import vegetation

# Constants
RHO_WATER = 1000.0
CP_SOIL = 800.0 # J/kg/K (approx for dry soil)
RHO_SOIL = 1600.0 # kg/m3
SOIL_DEPTH = 1.0 # meters (thermal depth) - Increased for stability
BUCKET_DEPTH = 0.15 # meters (water holding capacity)
L_VAPOR = 2.5e6

class LandState(NamedTuple):
    """State of the land surface."""
    temp: jnp.ndarray       # Surface temperature [K]
    soil_moisture: jnp.ndarray # Soil moisture depth [m] (0 to BUCKET_DEPTH)
    lai: jnp.ndarray        # Leaf Area Index [m2/m2]
    soil_carbon: jnp.ndarray # Soil Carbon [kg C/m2]

def init_land_state(ny, nx) -> LandState:
    """Initialize land state."""
    return LandState(
        temp=jnp.ones((ny, nx)) * 290.0,
        soil_moisture=jnp.ones((ny, nx)) * (BUCKET_DEPTH * 0.5), # 50% saturation
        lai=jnp.ones((ny, nx)) * 1.0, # Initial sparse vegetation
        soil_carbon=jnp.ones((ny, nx)) * 10.0 # 10 kg C/m2
    )

@partial(jax.jit, static_argnames=['ny', 'nx'])
def step_land(
    state: LandState,
    t_air: jnp.ndarray,    # Air temp [K]
    q_air: jnp.ndarray,    # Specific humidity [kg/kg]
    sw_down: jnp.ndarray,  # Shortwave down [W/m2]
    lw_down: jnp.ndarray,  # Longwave down [W/m2]
    precip: jnp.ndarray,   # Precipitation [m/s]
    mask: jnp.ndarray,     # Land mask (0=Ocean, 1=Land) - Note: Inverted from Ocean mask?
                           # Let's assume input is 1=Land, 0=Ocean for clarity here.
                           # Caller must ensure correct mask is passed.
    ny: int = ATMOS_GRID.nlat,
    nx: int = ATMOS_GRID.nlon
) -> Tuple[LandState, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Time step the land model.
    
    Args:
        state: Current land state
        t_air: Air temperature [K]
        q_air: Specific humidity [kg/kg]
        sw_down: Shortwave downward flux
        lw_down: Longwave downward flux
        precip: Precipitation rate [m/s]
        mask: Land mask (1=Land, 0=Ocean)
        
    Returns:
        new_state: Updated land state
        fluxes: (sensible_heat_flux, latent_heat_flux, carbon_flux)
                carbon_flux: [kg C/m2/s] (Positive Upward = Release to Atmos)
    """
    
    # 0. Update Vegetation
    lai_new = vegetation.step_vegetation(
        state.lai, state.temp, state.soil_moisture, BUCKET_DEPTH
    )
    
    # Get dynamic properties
    albedo, veg_fraction = vegetation.compute_land_properties(lai_new)
    
    # 1. Surface Energy Balance
    # Rn = SW_net + LW_net
    # SW_net = SW_down * (1 - alpha)
    sw_net = sw_down * (1.0 - albedo)
    
    # LW_net = LW_down - epsilon * sigma * T^4
    lw_up = EMISSIVITY_LAND * STEFAN_BOLTZMANN * state.temp**4
    lw_net = lw_down - lw_up
    
    r_net = sw_net + lw_net
    
    # Turbulent Fluxes (Bulk Aerodynamic)
    # H = rho * Cp * Ch * U * (Ts - Ta)
    # LE = rho * Lv * Ce * U * beta * (qs(Ts) - qa)
    
    # Simplified drag/transfer coeffs
    # Assume U = 5 m/s constant for now or pass wind
    wind_speed = 5.0 
    ch = DRAG_COEFF_LAND
    ce = DRAG_COEFF_LAND
    rho_air = 1.225
    cp_air = 1004.0
    
    # Sensible Heat (Positive Upward)
    sensible_flux = rho_air * cp_air * ch * wind_speed * (state.temp - t_air)
    
    # Latent Heat
    # Saturation humidity at Ts
    t_c = state.temp - 273.15
    es = 611.0 * jnp.exp(17.67 * t_c / (t_c + 243.5))
    p_surf = 101325.0
    q_sat = 0.622 * es / p_surf
    
    # Beta factor (Soil Moisture limitation)
    # beta = W / W_crit
    bucket_beta = state.soil_moisture / BUCKET_DEPTH
    bucket_beta = jnp.clip(bucket_beta, 0.0, 1.0)
    
    # Vegetation boosts evaporation (transpiration)
    beta = bucket_beta * (1.0 + veg_fraction)
    beta = jnp.clip(beta, 0.0, 1.0)
    
    # Potential Evaporation
    evap_pot = rho_air * ce * wind_speed * (q_sat - q_air)
    evap_pot = jnp.maximum(evap_pot, 0.0) # Only evaporation, no condensation for now
    
    # Actual Evaporation [kg/m2/s]
    evap = beta * evap_pot
    
    latent_flux = L_VAPOR * evap
    
    # Net Flux into Soil (Positive Warming)
    # G = Rn - H - LE
    g_flux = r_net - sensible_flux - latent_flux
    
    # 2. Temperature Update (Semi-Implicit)
    # Explicit Euler is unstable for thin soil / long timesteps.
    # We use a linearized implicit update:
    # C * (T_new - T_old) / dt = G(T_old) + dG/dT * (T_new - T_old)
    # T_new = T_old + G(T_old) / (C/dt - dG/dT)
    
    # Calculate derivatives (dG/dT = dRn/dT - dH/dT - dLE/dT)
    # All derivatives are negative (stabilizing). We want positive magnitude.
    
    # dRn/dT = -4 * epsilon * sigma * T^3
    d_rn_dt = -4 * EMISSIVITY_LAND * STEFAN_BOLTZMANN * state.temp**3
    
    # dH/dT = -rho * cp * Ch * U
    d_h_dt = -rho_air * cp_air * ch * wind_speed
    
    # dLE/dT = -rho * Lv * Ce * U * beta * dqs/dT
    # dqs/dT = qs * (17.67 * 243.5) / (Tc + 243.5)^2
    d_qs_dt = q_sat * (17.67 * 243.5) / ((t_c + 243.5)**2)
    d_le_dt = -rho_air * L_VAPOR * ce * wind_speed * beta * d_qs_dt
    
    # Total feedback (negative)
    d_g_dt = d_rn_dt + d_h_dt + d_le_dt
    
    # Heat Capacity
    heat_capacity = RHO_SOIL * CP_SOIL * SOIL_DEPTH
    
    # Implicit Update
    # (C/dt - dG/dT) * delta_T = G
    denominator = (heat_capacity / DT_ATMOS) - d_g_dt
    delta_temp = g_flux / denominator
    
    # Apply update only on land
    temp_new = state.temp + delta_temp * mask
    
    # Safety Clamping (Prevent runaway)
    temp_new = jnp.clip(temp_new, 180.0, 340.0)
    
    # Relax ocean points to T_air or keep constant? 
    # Keep constant (masked out in main) or sync with T_air to avoid NaNs
    temp_new = jnp.where(mask > 0.5, temp_new, state.temp)
    
    # 3. Soil Moisture Update
    # dW/dt = P - E - Runoff
    # P [m/s], E [kg/m2/s] -> [m/s] = E / rho_water
    
    evap_rate = evap / RHO_WATER
    
    dw_dt = precip - evap_rate
    
    moisture_new = state.soil_moisture + DT_ATMOS * dw_dt * mask
    
    # Bucket overflow (Runoff)
    runoff = jnp.maximum(moisture_new - BUCKET_DEPTH, 0.0)
    moisture_new = jnp.minimum(moisture_new, BUCKET_DEPTH)
    
    # Lower bound
    moisture_new = jnp.maximum(moisture_new, 0.0)
    
    # Mask ocean
    moisture_new = jnp.where(mask > 0.5, moisture_new, state.soil_moisture)
    
    # Mask ocean for LAI too
    lai_new = jnp.where(mask > 0.5, lai_new, state.lai)
    
    # 4. Carbon Cycle
    # GPP (Gross Primary Productivity) [kg C/m2/s]
    # Simple Light Use Efficiency model: GPP = LUE * SW_down * f_veg * f(T) * f(W)
    # We already have growth_potential in vegetation.py, but let's simplify here.
    # GPP proportional to LAI and SW_down.
    lue = 1.0e-9 # Light Use Efficiency (very approx)
    gpp = lue * sw_down * lai_new * beta # beta is moisture stress
    
    # Respiration
    # R_auto (Plant) ~ 0.5 * GPP
    r_auto = 0.5 * gpp
    
    # R_hetero (Soil) ~ k * SoilCarbon * f(T) * f(W)
    # Q10 = 2.0
    t_ref = 288.15
    q10_factor = 2.0 ** ((state.temp - t_ref) / 10.0)
    k_soil = 1.0e-8 # Decay rate [1/s]
    r_hetero = k_soil * state.soil_carbon * q10_factor * beta
    
    # Net Ecosystem Exchange (NEE) = Respiration - GPP
    # Positive = Release to Atmos
    nee = r_auto + r_hetero - gpp
    
    # Soil Carbon Update
    # Input: Litterfall (turnover of veg). Assume steady state veg biomass for now?
    # Or assume a fraction of GPP goes to soil eventually.
    # dC_soil/dt = Litter - R_hetero
    # Assume Litter ~ GPP - R_auto (NPP) eventually dies and becomes soil C
    litter = gpp - r_auto
    
    d_soil_c_dt = litter - r_hetero
    soil_carbon_new = state.soil_carbon + DT_ATMOS * d_soil_c_dt * mask
    soil_carbon_new = jnp.maximum(soil_carbon_new, 0.0)
    
    # Mask ocean
    soil_carbon_new = jnp.where(mask > 0.5, soil_carbon_new, state.soil_carbon)
    
    new_state = LandState(
        temp=temp_new,
        soil_moisture=moisture_new,
        lai=lai_new,
        soil_carbon=soil_carbon_new
    )
    
    return new_state, (sensible_flux, latent_flux, nee)
