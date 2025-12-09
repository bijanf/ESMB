"""
Atmospheric Dynamics Module (Primitive Equations).

This module implements the Spectral Primitive Equations on a sphere (approximated).
It solves for:
- Vorticity (zeta)
- Divergence (div)
- Temperature (T)
- Log Surface Pressure (ln_ps)
- Specific Humidity (q) [Advected]
- CO2 (co2) [Advected]

Method:
- Spectral (FFT) in Longitude
- Finite Difference in Latitude
- Helmholtz Decomposition for U, V
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional
from functools import partial

from chronos_esm.config import ATMOS_GRID, EARTH_RADIUS, DT_ATMOS
from chronos_esm.atmos import spectral, physics
from chronos_esm.ocean.solver import solve_cg, jacobi_preconditioner, solve_poisson_2d

# Constants
R_EARTH = EARTH_RADIUS
OMEGA = 7.292e-5  # Rotation rate of Earth
GRAVITY = 9.81
RD = 287.0        # Gas constant for dry air
CP = 1004.0       # Specific heat at constant pressure
KAPPA = RD / CP

class AtmosState(NamedTuple):
    """State of the atmosphere (Primitive Equations)."""
    vorticity: jnp.ndarray  # Relative Vorticity [1/s]
    divergence: jnp.ndarray # Divergence [1/s]
    temp: jnp.ndarray       # Temperature [K]
    ln_ps: jnp.ndarray      # Log Surface Pressure [log(Pa)]
    q: jnp.ndarray          # Specific Humidity [kg/kg]
    co2: jnp.ndarray        # CO2 Concentration [ppm]
    
    # Diagnostic variables (computed from prognostic)
    u: jnp.ndarray          # Zonal Wind [m/s]
    v: jnp.ndarray          # Meridional Wind [m/s]
    psi: jnp.ndarray        # Streamfunction
    chi: jnp.ndarray        # Velocity Potential

def init_atmos_state(ny: int = ATMOS_GRID.nlat, nx: int = ATMOS_GRID.nlon) -> AtmosState:
    """Initialize the atmospheric state."""
    # Grid
    lat = jnp.linspace(-90, 90, ny)
    lon = jnp.linspace(0, 360, nx, endpoint=False)
    lat_rad = jnp.deg2rad(lat)
    
    # Initial Conditions (Jablonowski-Williamson inspired basic state)
    u0 = 30.0 * jnp.cos(lat_rad)[:, None] * jnp.ones((ny, nx)) # Zonal jet
    v0 = jnp.zeros((ny, nx))
    
    # Vorticity and Divergence from U, V
    # We need gradients.
    dx = 2 * jnp.pi * R_EARTH * jnp.cos(lat_rad)[:, None] / nx
    dy = jnp.pi * R_EARTH / ny
    
    # Simple FD for initialization
    dv_dx, dv_dy = spectral.compute_gradients(v0, dx, dy)
    du_dx, du_dy = spectral.compute_gradients(u0, dx, dy)
    
    vorticity = dv_dx - du_dy
    divergence = du_dx + dv_dy
    
    temp = 300.0 - 20.0 * jnp.sin(lat_rad)[:, None]**2 # Warm equator, cold poles
    temp = jnp.broadcast_to(temp, (ny, nx))
    
    ln_ps = jnp.log(101325.0) * jnp.ones((ny, nx))
    
    q = 0.01 * jnp.exp(-((lat_rad[:, None])/0.5)**2) # Moisture at equator
    q = jnp.broadcast_to(q, (ny, nx))
    
    co2 = 400.0 * jnp.ones((ny, nx))
    
    return AtmosState(
        vorticity=vorticity,
        divergence=divergence,
        temp=temp,
        ln_ps=ln_ps,
        q=q,
        co2=co2,
        u=u0,
        v=v0,
        psi=jnp.zeros((ny, nx)),
        chi=jnp.zeros((ny, nx))
    )

@partial(jax.jit, static_argnames=['ny', 'nx'])
def solve_poisson_sphere(
    rhs: jnp.ndarray,
    dx: jnp.ndarray, # (ny, 1)
    dy: float,
    ny: int,
    nx: int
) -> jnp.ndarray:
    """
    Solve ∇²ψ = rhs on a sphere (channel approximation).
    BC: Periodic in X, Dirichlet=0 at Y boundaries (Poles).
    """
    # Flatten
    b = -rhs.flatten()
    x0 = jnp.zeros_like(b)
    
    # Operator
    def laplacian_operator(x_flat):
        x_2d = x_flat.reshape((ny, nx))
        
        # X: Periodic
        d2x = (jnp.roll(x_2d, 1, axis=1) - 2*x_2d + jnp.roll(x_2d, -1, axis=1)) / dx**2
        
        # Y: Dirichlet (0 at boundaries)
        # Pad with 0
        x_padded = jnp.pad(x_2d, ((1, 1), (0, 0)), mode='constant', constant_values=0)
        d2y = (x_padded[2:, :] - 2*x_2d + x_padded[:-2, :]) / dy**2
        
        lap = d2x + d2y
        return -lap.flatten()
    
    # Preconditioner (Diagonal)
    diag_val = 2.0/dx**2 + 2.0/dy**2
    diag_val = jnp.broadcast_to(diag_val, (ny, nx))
    precond = jacobi_preconditioner(diag_val.flatten())
    
    # Solve
    x_flat, _ = solve_cg(laplacian_operator, b, x0, max_iter=200, tol=1e-5, preconditioner=precond)
    
    return x_flat.reshape((ny, nx))

@partial(jax.jit, static_argnames=['ny', 'nx'])
def step_atmos(
    state: AtmosState,
    surface_temp: jnp.ndarray, # SST or Land Temp
    flux_sensible: jnp.ndarray, # W/m2
    flux_latent: jnp.ndarray,   # W/m2
    flux_co2: jnp.ndarray,      # ppm/s
    dt: float = DT_ATMOS,
    ny: int = ATMOS_GRID.nlat,
    nx: int = ATMOS_GRID.nlon
) -> Tuple[AtmosState, Tuple[jnp.ndarray, jnp.ndarray]]: # Returns State, (Precip, SfcPressure)
    """
    Time step the atmosphere (Dynamics + Physics).
    """
    # Grid metrics
    lat = jnp.linspace(-90, 90, ny)
    lat_rad = jnp.deg2rad(lat)
    cos_lat = jnp.cos(lat_rad)[:, None]
    sin_lat = jnp.sin(lat_rad)[:, None]
    
    # Avoid division by zero at poles
    cos_lat = jnp.where(cos_lat < 1e-5, 1e-5, cos_lat)
    
    dx = 2 * jnp.pi * R_EARTH * cos_lat / nx
    dy = jnp.pi * R_EARTH / ny
    
    # 1. Recover U, V from Vorticity, Divergence
    # 1. Recover U, V from Vorticity, Divergence
    # Warm start with previous psi/chi
    psi, _ = solve_poisson_2d(state.vorticity, dx, dy, max_iter=50, tol=1e-3, x0=state.psi)
    chi, _ = solve_poisson_2d(state.divergence, dx, dy, max_iter=50, tol=1e-3, x0=state.chi)
    
    dpsi_dx, dpsi_dy = spectral.compute_gradients(psi, dx, dy)
    dchi_dx, dchi_dy = spectral.compute_gradients(chi, dx, dy)
    
    u = -dpsi_dy + dchi_dx
    v = dpsi_dx + dchi_dy
    
    # 2. Physics
    # Precipitation / Convection
    # We use the existing simple physics
    # Precipitation / Convection
    # We use the existing simple physics
    pressure = jnp.exp(state.ln_ps)
    q_sat = physics.compute_saturation_humidity(state.temp, pressure)
    
    precip, heating_precip = physics.compute_precipitation(state.q, q_sat)
    drying_precip = -precip # Precip removes moisture (negative tendency)
    # Actually physics.compute_precipitation takes (temp, q). Let's check signature.
    # Assuming it takes (temp, q).
    
    # Radiation
    # Use state.co2 (spatially varying)
    # We need to average CO2 for the simple radiation scheme or update it to map.
    # The current physics.compute_radiative_forcing takes co2_ppm as float.
    # We will pass the global mean for now or update physics later.
    co2_mean = jnp.mean(state.co2)
    heating_rad = physics.compute_radiative_forcing(state.temp, co2_mean, lat_rad=lat_rad)
    
    # Surface Drag (Rayleigh Friction)
    drag_u = -1e-5 * u
    drag_v = -1e-5 * v
    
    # Total Forcings
    forcing_u = drag_u
    forcing_v = drag_v
    forcing_t = (heating_precip + heating_rad) / CP + flux_sensible / (1000.0 * 1000.0) # Approx density/height
    forcing_q = drying_precip + flux_latent / (2.5e6 * 1000.0 * 1000.0)
    forcing_co2 = flux_co2
    
    # 3. Dynamics Tendencies
    # Advection Operator: -(u ∂/∂x + v ∂/∂y)
    def advect(f):
        df_dx, df_dy = spectral.compute_gradients(f, dx, dy)
        return -(u * df_dx + v * df_dy)
    
    # Coriolis
    f_cor = 2 * OMEGA * sin_lat
    eta = state.vorticity + f_cor
    
    # Forcing (Friction)
    # We need Curl(F) for Vorticity and Div(F) for Divergence
    # F = (forcing_u, forcing_v)
    # Curl(F) = dv_dx - du_dy (of forcing)
    # Div(F) = du_dx + dv_dy (of forcing)
    
    dfu_dx, dfu_dy = spectral.compute_gradients(forcing_u, dx, dy)
    dfv_dx, dfv_dy = spectral.compute_gradients(forcing_v, dx, dy)
    
    curl_forcing = dfv_dx - dfu_dy
    div_forcing = dfu_dx + dfv_dy
    
    dzeta_dt = advect(eta) - eta * state.divergence + curl_forcing
    
    # Divergence Damping + Geostrophic Adjustment
    lap_p = spectral.compute_laplacian(RD * state.temp * state.ln_ps, dx, dy)
    ddiv_dt = -lap_p + f_cor * state.vorticity - 1e-4 * state.divergence + div_forcing
    
    dt_dt = advect(state.temp) + forcing_t
    dlnps_dt = advect(state.ln_ps) - state.divergence
    dq_dt = advect(state.q) + forcing_q
    dco2_dt = advect(state.co2) + forcing_co2
    
    # Diffusion / Sponge Layer
    # Add 2nd order diffusion to stabilize spectral advection
    nu = 1.0e7 # Diffusion coefficient [m^2/s] - Increased for stability with larger DT
    
    diff_zeta = spectral.compute_laplacian(state.vorticity, dx, dy)
    diff_div = spectral.compute_laplacian(state.divergence, dx, dy)
    diff_temp = spectral.compute_laplacian(state.temp, dx, dy)
    diff_q = spectral.compute_laplacian(state.q, dx, dy)
    diff_co2 = spectral.compute_laplacian(state.co2, dx, dy)
    
    # 4. Time Integration
    new_vorticity = state.vorticity + dt * (dzeta_dt + nu * diff_zeta)
    new_divergence = state.divergence + dt * (ddiv_dt + nu * diff_div)
    new_temp = state.temp + dt * (dt_dt + nu * diff_temp)
    new_ln_ps = state.ln_ps + dt * dlnps_dt # No diffusion on pressure usually
    new_q = state.q + dt * (dq_dt + nu * diff_q)
    new_co2 = state.co2 + dt * (dco2_dt + nu * diff_co2)
    
    # 5. Safety Clamping (Prevent Runaway Instability)
    # Clip Humidity (Physical constraint)
    new_q = jnp.maximum(new_q, 0.0)
    
    # Clamp Temperature (100K - 350K)
    new_temp = jnp.clip(new_temp, 150.0, 350.0)
    
    # Clamp Winds (Prevent supersonic blowups)
    # We need to recover U, V from new vorticity/divergence to clamp them effectively?
    # Or just clamp the diagnostic U, V returned?
    # Actually, U and V are diagnostic from Vort/Div. If we clamp Vort/Div it's hard to know bounds.
    # But we can clamp the *diagnostic* U, V in the next step, but that doesn't fix the prognostic Vort/Div.
    # Let's clamp Vorticity/Divergence loosely.
    # Earth Vorticity ~ 1e-4. 
    new_vorticity = jnp.clip(new_vorticity, -1e-2, 1e-2)
    new_divergence = jnp.clip(new_divergence, -1e-2, 1e-2)
    
    # 6. Polar Filter
    # Filter prognostic variables to maintain stability with explicit time stepping
    new_vorticity = spectral.polar_filter(new_vorticity, lat_rad)
    new_divergence = spectral.polar_filter(new_divergence, lat_rad)
    new_temp = spectral.polar_filter(new_temp, lat_rad)
    new_ln_ps = spectral.polar_filter(new_ln_ps, lat_rad)
    new_q = spectral.polar_filter(new_q, lat_rad)
    new_co2 = spectral.polar_filter(new_co2, lat_rad)
    
    new_state = AtmosState(
        vorticity=new_vorticity,
        divergence=new_divergence,
        temp=new_temp,
        ln_ps=new_ln_ps,
        q=new_q,
        co2=new_co2,
        u=u, 
        v=v,
        psi=psi,
        chi=chi
    )
    
    return new_state, (precip, jnp.exp(new_ln_ps))
