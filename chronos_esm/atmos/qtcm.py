"""
QTCM (Quasi-Equilibrium Tropical Circulation Model) Stepper.

Main driver for the atmospheric component.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from functools import partial

from chronos_esm.config import ATMOS_GRID, DT_ATMOS
from chronos_esm.atmos import spectral, physics


class AtmosState(NamedTuple):
    """State of the atmosphere."""
    u: jnp.ndarray       # Zonal velocity (ny, nx) - Barotropic/First baroclinic
    v: jnp.ndarray       # Meridional velocity
    temp: jnp.ndarray    # Temperature (ny, nx)
    q: jnp.ndarray       # Specific humidity (ny, nx)
    co2: jnp.ndarray     # CO2 concentration [ppm] (ny, nx)
    
    
def init_atmos_state(ny, nx, co2_ppm=280.0) -> AtmosState:
    """Initialize atmosphere state."""
    temp = jnp.ones((ny, nx)) * 280.0
    
    # Initialize q to 80% RH to avoid latent heat shock
    t_c = temp - 273.15
    es = 611.0 * jnp.exp(17.67 * t_c / (t_c + 243.5))
    p_surf = 101325.0
    q_sat = 0.622 * es / p_surf
    q = q_sat * 0.8
    
    return AtmosState(
        u=jnp.zeros((ny, nx)),
        v=jnp.zeros((ny, nx)),
        temp=temp,
        q=q,
        co2=jnp.ones((ny, nx)) * co2_ppm
    )


@partial(jax.jit, static_argnames=['ny', 'nx'])
def step_atmos(
    state: AtmosState,
    sst: jnp.ndarray,
    co2_flux: jnp.ndarray, # CO2 flux from surface [ppm/s] (Positive Upward)
    dx: float,
    dy: float,
    ny: int = ATMOS_GRID.nlat,
    nx: int = ATMOS_GRID.nlon,
    beta: jnp.ndarray = None # Evaporation efficiency (0-1)
) -> Tuple[AtmosState, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Time step the atmosphere.
    
    1. Dynamics (Simplified Vorticity/Divergence or Primitive Eq)
    2. Physics (Precipitation, Radiation)
    3. Coupling Fluxes
    
    Args:
        state: Current state
        sst: Sea Surface Temperature [K]
        co2_flux: CO2 flux from surface [ppm/s]
        dx, dy: Grid spacing
        beta: Evaporation efficiency (1.0=Ocean, 0-1=Land)
        
    Returns:
        new_state: Updated state
        fluxes: (heat_flux, fw_flux) to ocean
    """
    # Default beta to 1.0 (wet surface) if not provided
    if beta is None:
        beta = jnp.ones_like(sst)
        
    # 1. Physics
    # Saturation humidity (Clausius-Clapeyron approx)
    # es = 611 * exp(17.67 * (T - 273.15) / (T - 29.65))
    t_c = state.temp - 273.15
    es = 611.0 * jnp.exp(17.67 * t_c / (t_c + 243.5))
    p_surf = 101325.0
    q_sat = 0.622 * es / p_surf
    
    # Precipitation & Heating
    precip, latent_heat = physics.compute_precipitation(state.q, q_sat)
    
    # Radiative Cooling & CO2 Forcing
    rad_heating = physics.compute_radiative_forcing(state.temp, state.co2)
    
    # 2. Dynamics (Simplified Advection)
    # dT/dt = -u.grad(T) + Heating
    # dq/dt = -u.grad(q) - Precip + Evap
    
    # Gradients
    dt_dx, dt_dy = spectral.compute_gradients(state.temp, dx, dy)
    dq_dx, dq_dy = spectral.compute_gradients(state.q, dx, dy)
    dco2_dx, dco2_dy = spectral.compute_gradients(state.co2, dx, dy)
    
    # Advection terms
    adv_t = -(state.u * dt_dx + state.v * dt_dy)
    adv_q = -(state.u * dq_dx + state.v * dq_dy)
    adv_co2 = -(state.u * dco2_dx + state.v * dco2_dy)
    
    # Surface Fluxes (Bulk Aerodynamic Formula)
    # Evaporation E = beta * C_d * |U| * (q_sat(SST) - q)
    # Sensible Heat H = C_d * |U| * (SST - T_air)
    
    wind_speed = jnp.sqrt(state.u**2 + state.v**2 + 1.0) # +1 gustiness
    c_d = 1.5e-3 # Drag coefficient
    
    # q_sat at SST
    sst_c = sst - 273.15
    es_sst = 611.0 * jnp.exp(17.67 * sst_c / (sst_c + 243.5))
    q_sat_sst = 0.622 * es_sst / p_surf
    
    evap = beta * c_d * wind_speed * (q_sat_sst - state.q)
    sensible = c_d * wind_speed * (sst - state.temp)
    
    # 3. Time Integration
    # Temperature
    # dT/dt = Advection + Latent Heat + Radiation + Sensible (distributed)
    # Distribute sensible heat over boundary layer? Assume column average for QTCM.
    # H_scale ~ 8000m. Cp ~ 1000.
    sensible_heating = sensible / (10000.0 * 1004.0) # Scaling factor: 1 / (rho*H * Cp)
    
    temp_tendency = adv_t + latent_heat + rad_heating + sensible_heating
    
    # Cap tendency to prevent explosion (e.g. max 10 K/day ~ 1e-4 K/s)
    temp_tendency = jnp.clip(temp_tendency, -2e-4, 2e-4)
    
    # Bypass Physics for debugging NaNs
    # temp_tendency = jnp.zeros_like(state.temp)
    
    temp_new = state.temp + DT_ATMOS * temp_tendency
    
    # Moisture
    # dq/dt = Advection - Precip + Evap (distributed)
    evap_moistening = evap / 10000.0 # Scaling
    
    q_tendency = adv_q - precip + evap_moistening
    q_tendency = jnp.clip(q_tendency, -1e-6, 1e-6)
    
    q_new = state.q + DT_ATMOS * q_tendency
    
    # CO2
    # dCO2/dt = Advection + Flux
    # Flux is [ppm/s]
    co2_tendency = adv_co2 + co2_flux
    co2_new = state.co2 + DT_ATMOS * co2_tendency
    
    # Winds (Geostrophic / Thermal Wind Balance - Simplified)
    # In full QTCM, we solve momentum equations.
    # Here, let's just relax winds or keep them constant for this phase 
    # until we implement full spectral dynamics.
    # Or drive them by temperature gradients (thermal wind).
    u_new = state.u
    v_new = state.v
    
    new_state = AtmosState(
        u=u_new,
        v=v_new,
        temp=temp_new,
        q=q_new,
        co2=co2_new
    )
    
    # Fluxes to Ocean
    # Heat Flux = Net Solar - Net Longwave - Latent - Sensible
    # Simplified: Just return net downward heat flux
    # Latent flux [W/m^2] = L_v * Evap [kg/m^2/s]
    latent_flux = 2.5e6 * evap
    sensible_flux = 1004.0 * sensible # Check units: sensible calc was K-based? 
    # Wait, sensible in code above: c_d * U * (SST - T). Units: m/s * K.
    # Heat flux = rho * Cp * sensible_kinematic
    rho_air = 1.225
    sensible_heat_flux = rho_air * 1004.0 * sensible
    latent_heat_flux = rho_air * 2.5e6 * evap
    
    # Freshwater flux (Precip - Evap) [kg/m^2/s]
    # Precip is in kg/kg/s (rate). Convert to mass flux: rho * H * P
    # Assume P is column integrated? In compute_precip it was rate.
    # Let's assume precip is mass flux directly or scaled.
    # Re-check physics.py: precip = softplus * eps/tau. Units 1/s.
    # So it's specific humidity tendency.
    # Mass flux = Integral(rho * P) dz ~ rho * H * P
    precip_flux = rho_air * 8000.0 * precip
    evap_flux = rho_air * evap
    
    # Net heat flux into ocean (positive down)
    # Q_net = Q_solar - Q_lw_net - Q_lat - Q_sens
    
    # Solar (Shortwave)
    # Simple average insolation approx 340 W/m^2 * (1 - albedo)
    # Albedo ocean ~ 0.06
    # We should vary this with latitude, but for now constant mean.
    q_solar = 240.0 # W/m^2 (approx global mean absorbed)
    
    # Longwave (Net Upward)
    # sigma * T^4 - DownwardLW
    # Net cooling approx 50-70 W/m^2
    q_lw_net = 60.0 # W/m^2
    
    # Total Flux
    net_heat_flux = q_solar - q_lw_net - sensible_heat_flux - latent_heat_flux
    
    # Flux Limiter (Safety)
    # Cap fluxes to prevent numerical explosions
    net_heat_flux = jnp.clip(net_heat_flux, -1000.0, 1000.0)
    fw_flux = jnp.clip(precip_flux - evap_flux, -0.01, 0.01) # kg/m^2/s
    
    fw_flux = precip_flux - evap_flux
    
    return new_state, (net_heat_flux, fw_flux)
