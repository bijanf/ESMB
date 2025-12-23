"""
JAX-based driver for the Ocean component (Veros-like physics).
"""

from functools import partial
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from chronos_esm.config import (  # noqa: F401
    DT_OCEAN,
    GRAVITY,
    OCEAN_GRID,
    OMEGA,
    RHO_WATER,
)
from chronos_esm.ocean import mixing, solver


class OceanState(NamedTuple):
    """State of the ocean model."""

    u: jnp.ndarray  # Zonal velocity (nz, ny, nx)
    v: jnp.ndarray  # Meridional velocity (nz, ny, nx)
    w: jnp.ndarray  # Vertical velocity (nz, ny, nx)
    temp: jnp.ndarray  # Potential temperature (nz, ny, nx)
    salt: jnp.ndarray  # Salinity (nz, ny, nx)
    psi: jnp.ndarray  # Barotropic streamfunction (ny, nx)
    rho: jnp.ndarray  # Density (nz, ny, nx)
    dic: jnp.ndarray  # Dissolved Inorganic Carbon [mmol/m^3] (nz, ny, nx)


def equation_of_state(temp: jnp.ndarray, salt: jnp.ndarray) -> jnp.ndarray:
    """
    Linear equation of state for seawater.
    rho = rho_0 - alpha * (T - T0) + beta * (S - S0)
    """
    rho_0 = RHO_WATER
    alpha = 0.2  # Thermal expansion coeff [kg/m^3/K]
    beta = 0.8  # Haline contraction coeff [kg/m^3/ppt]
    t0 = 283.15  # Reference T (10 C in Kelvin)
    s0 = 35.0

    return rho_0 - alpha * (temp - t0) + beta * (salt - s0)


@partial(jax.jit, static_argnames=["nz", "ny", "nx", "dt"])
def step_ocean(
    state: OceanState,
    surface_fluxes: Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray
    ],  # (heat_flux, fresh_water_flux, dic_flux)
    wind_stress: Tuple[jnp.ndarray, jnp.ndarray],  # (tau_x, tau_y)
    dx: float,
    dy: float,
    dz: jnp.ndarray,
    nz: int = OCEAN_GRID.nz,
    ny: int = OCEAN_GRID.nlat,
    nx: int = OCEAN_GRID.nlon,
    mask: Optional[jnp.ndarray] = None,  # 1=Ocean, 0=Land
    dt: float = DT_OCEAN,
    r_drag: float = 5.0e-2,
    kappa_gm: float = 1000.0,
) -> OceanState:
    """
    Time step the ocean model.

    1. Update density
    2. Calculate mixing (GM)
    3. Solve barotropic streamfunction
    4. Update velocity (momentum)
    5. Update tracers (T, S)
    """

    # 1. Update Density
    rho = equation_of_state(state.temp, state.salt)

    # 2. Mixing Parameters
    # Calculate isopycnal slopes and GM velocities
    sx, sy = mixing.compute_isopycnal_slopes(rho, dx, dy, dz)

    # GM Diffusivity (constant for now, could be flow dependent)
    # kappa_gm = 1000.0 # NOW PASSED AS ARGUMENT
    u_bolus, v_bolus, _ = mixing.compute_gm_bolus_velocity(kappa_gm, sx, sy, dz)

    # Effective transport velocity = Eulerian + Bolus
    u_eff = state.u + u_bolus
    v_eff = state.v + v_bolus

    # 3. Barotropic Streamfunction Solver
    # ∇ · (H^{-1} ∇ψ) = curl(tau) / rho0 + ... (vorticity forcing)
    # For this simplified step, we'll solve a Poisson-like equation for psi
    # driven by wind stress curl.

    # Calculate Wind Stress Curl
    tau_x, tau_y = wind_stress
    curl_tau = (jnp.roll(tau_y, -1, axis=1) - jnp.roll(tau_y, 1, axis=1)) / (2 * dx) - (
        jnp.roll(tau_x, -1, axis=0) - jnp.roll(tau_x, 1, axis=0)
    ) / (2 * dy)

    # RHS for streamfunction solver
    # In full Veros, this includes baroclinic torque, etc.
    # Here: simplified barotropic vorticity equation
    rhs_psi = curl_tau / (RHO_WATER * dz[0])  # Normalized roughly

    # Solve for psi
    # Using our CG solver
    # Pass mask to enforce no flow through land
    psi_new, _ = solver.solve_poisson_2d(
        rhs_psi, dx, dy, max_iter=50, tol=1e-3, mask=mask, x0=state.psi
    )

    # Update barotropic velocities from psi
    # u_bt = -∂ψ/∂y, v_bt = ∂ψ/∂x
    u_bt = -(jnp.roll(psi_new, -1, axis=0) - jnp.roll(psi_new, 1, axis=0)) / (2 * dy)
    v_bt = (jnp.roll(psi_new, -1, axis=1) - jnp.roll(psi_new, 1, axis=1)) / (2 * dx)

    # Broadcast to 3D
    u_bt_3d = jnp.broadcast_to(u_bt, (nz, ny, nx))
    v_bt_3d = jnp.broadcast_to(v_bt, (nz, ny, nx))

    # 4. Update Velocity (Barotropic + Baroclinic)
    # --------------------------------------------
    # We now have the Barotropic Flow (u_bt, v_bt) which is depth-independent.
    # We need to add the Baroclinic (Shear) component derived from Thermal Wind Balance.
    
    # A. Calculate Hydrostatic Pressure
    # P(z) = g * integral_z^0 rho dz'
    # We integrate from surface down.
    # P_hydro[0] = 0 (assuming rigid lid for pressure anomaly)
    # P_hydro[k] = P_hydro[k-1] + g * 0.5 * (rho[k-1] + rho[k]) * dz[k-1] ??
    # Let's use a simple accumulation.
    
    # Density anomaly from reference
    rho_anom = rho - RHO_WATER
    
    # Pressure integration (cumulative sum)
    # P at level k center can be approx: Sum_{i=0}^{k-1} (g * rho_i * dz_i) + g * rho_k * 0.5 * dz_k
    # Simplified: Cumsum density * dz * g
    
    # Broadcast dz to 3D
    dz_3d = dz.reshape(-1, 1, 1) # (nz, 1, 1)
    
    # Hydrostatic pressure (nz, ny, nx)
    # Note: We integrate DOWNWARDS.
    # For TWB, we need horizontal gradients of P.
    
    # P_hydro = Integral(-g * rho) dz.
    # Since z is negative downwards? 
    # Usually: dp/dz = -rho * g  => p(z) = Integral_z^0 (rho * g) dz'
    
    g_rho = GRAVITY * rho_anom
    pressure_hydro = jnp.cumsum(g_rho * dz_3d, axis=0)
    
    # B. Calculate Geostrophic Shear (Thermal Wind)
    # f * u = -1/rho0 * dp/dy
    # f * v =  1/rho0 * dp/dx
    
    # Coriolis Parameter (f)
    # lat from -90 to 90
    lat_deg = jnp.linspace(-90, 90, ny)
    lat_rad = jnp.deg2rad(lat_deg)
    omega = OMEGA
    f_coriolis = 2 * omega * jnp.sin(lat_rad)
    
    # Broadcast f to (nz, ny, nx)
    f_3d = jnp.broadcast_to(f_coriolis[None, :, None], (nz, ny, nx))
    
    # Avoid division by zero at equator (clamp f)
    f_min = 1e-5
    f_clamped = jnp.where(jnp.abs(f_3d) < f_min, jnp.sign(f_3d + 1e-16) * f_min, f_3d)
    
    # Gradients of Pressure
    # dp/dx
    dp_dx = (jnp.roll(pressure_hydro, -1, axis=2) - jnp.roll(pressure_hydro, 1, axis=2)) / (2 * dx)
    # dp/dy
    dp_dy = (jnp.roll(pressure_hydro, -1, axis=1) - jnp.roll(pressure_hydro, 1, axis=1)) / (2 * dy)
    
    # Geostrophic Velocity
    # Geostrophic Velocity with Rayleigh Friction
    # r*u - f*v = -1/rho * px
    # f*u + r*v = -1/rho * py
    # Solution:
    # u = -(r*px + f*py) / (rho * (f^2 + r^2))
    # v =  (f*px - r*py) / (rho * (f^2 + r^2))
    
    # r_drag used from function argument
    
    denom = RHO_WATER * (f_clamped**2 + r_drag**2)
    
    u_geo = -(r_drag * dp_dx + f_clamped * dp_dy) / denom
    v_geo =  (f_clamped * dp_dx - r_drag * dp_dy) / denom
    
    # Damping at equator to prevent explosion
    # Simple mask: if |lat| < 5 deg, damp velocities
    equator_mask = jnp.abs(lat_deg) < 5.0
    damp_factor = jnp.where(equator_mask[None, :, None], 0.0, 1.0)
    
    u_geo = u_geo * damp_factor
    v_geo = v_geo * damp_factor
    
    # C. Baroclinic Anomaly
    # The barotropic solver gives the vertically averaged flow.
    # We must ensure the baroclinic component has ZERO vertical average.
    # u_bc = u_geo - mean(u_geo)
    
    u_geo_mean = jnp.sum(u_geo * dz_3d, axis=0) / jnp.sum(dz_3d, axis=0) # (ny, nx)
    v_geo_mean = jnp.sum(v_geo * dz_3d, axis=0) / jnp.sum(dz_3d, axis=0) # (ny, nx)
    
    u_bc = u_geo - u_geo_mean[None, :, :]
    v_bc = v_geo - v_geo_mean[None, :, :]
    
    # D. Total Velocity
    u_new = u_bt_3d + u_bc
    v_new = v_bt_3d + v_bc
    
    # Vertical velocity w (from continuity)
    # div(u) + dw/dz = 0 => dw/dz = -div(u)
    # w(z) = - Integral_bottom^z div(u) dz'
    # We won't compute full w here, stick to old w for now or zero?
    # Keeping old w is safer for stability for now, or assume w=0
    w_new = state.w 

    # 5. Update Tracers (Advection-Diffusion)
    # Simple upwind advection + vertical diffusion
    # dT/dt = - (u_eff ∂T/∂x + v_eff ∂T/∂y) + F_surf

    heat_flux, fw_flux, dic_flux = surface_fluxes

    # Advection (Upwind simplified)
    def advect(field, u, v):
        # x-flux
        f_x = jnp.where(u > 0, u * jnp.roll(field, 1, axis=2), u * field)
        adv_x = (f_x - jnp.roll(f_x, 1, axis=2)) / dx

        # y-flux
        f_y = jnp.where(v > 0, v * jnp.roll(field, 1, axis=1), v * field)
        adv_y = (f_y - jnp.roll(f_y, 1, axis=1)) / dy

        return -(adv_x + adv_y)

    dT_adv = advect(state.temp, u_eff, v_eff)
    dT_adv = advect(state.temp, u_eff, v_eff)
    dS_adv = advect(state.salt, u_eff, v_eff)
    dDIC_adv = advect(state.dic, u_eff, v_eff)

    # Surface forcing (applied to top layer k=0)
    # Heat flux [W/m^2] -> [K/s] : Q / (rho * cp * dz)
    # FW flux [kg/m^2/s] -> [ppt/s] : ... simplified

    forcing_T = jnp.zeros_like(state.temp)
    forcing_S = jnp.zeros_like(state.salt)
    forcing_DIC = jnp.zeros_like(state.dic)

    # Apply to surface layer
    cp = 3985.0

    # Thermal Inertia Scaling for Land
    # If mask is provided (1=Ocean, 0=Land)
    # Land has much lower heat capacity. Simulate by reducing effective depth.
    # Ocean dz[0] ~ 100m. Land effective ~ 2m. Factor ~ 50.
    # REVERTED: Now handled by dedicated Land Model.
    # if mask is not None:
    #     # mask is (ny, nx). Broadcast to top layer shape if needed.
    #     # heat_cap_factor = 1.0 on Ocean, 0.02 on Land
    #     heat_cap_factor = mask + (1.0 - mask) * 0.02
    #     effective_depth = dz[0] * heat_cap_factor
    # else:
    #     effective_depth = dz[0]

    forcing_T = forcing_T.at[0].set(heat_flux / (RHO_WATER * cp * dz[0]))
    forcing_S = forcing_S.at[0].set(
        fw_flux * 35.0 / (RHO_WATER * dz[0])
    )  # Simplified salinity flux
    forcing_DIC = forcing_DIC.at[0].set(
        dic_flux / dz[0]
    )  # Flux [mmol/m2/s] -> [mmol/m3/s]

    forcing_DIC = forcing_DIC.at[0].set(
        dic_flux / dz[0]
    )  # Flux [mmol/m2/s] -> [mmol/m3/s]

    # Time integration (Euler)
    temp_new = state.temp + dt * (dT_adv + forcing_T)
    salt_new = state.salt + dt * (dS_adv + forcing_S)
    dic_new = state.dic + dt * (dDIC_adv + forcing_DIC)

    # Update Density
    rho_new = equation_of_state(temp_new, salt_new)

    # Update State
    # Safety Clamping
    temp_new = jnp.clip(temp_new, -3.0 + 273.15, 50.0 + 273.15)
    salt_new = jnp.clip(salt_new, 0.0, 50.0)

    new_state = OceanState(
        u=u_new,
        v=v_new,
        w=w_new,
        temp=temp_new,
        salt=salt_new,
        psi=psi_new,
        rho=rho_new,
        dic=dic_new,
    )

    return new_state


def init_ocean_state(nz, ny, nx) -> OceanState:
    """Initialize a resting ocean state."""
    return OceanState(
        u=jnp.zeros((nz, ny, nx)),
        v=jnp.zeros((nz, ny, nx)),
        w=jnp.zeros((nz, ny, nx)),
        temp=jnp.ones((nz, ny, nx)) * (10.0 + 273.15),  # 10 deg C in Kelvin
        salt=jnp.ones((nz, ny, nx)) * 35.0,  # 35 ppt
        psi=jnp.zeros((ny, nx)),
        rho=jnp.zeros((nz, ny, nx)),
        dic=jnp.ones((nz, ny, nx)) * 2000.0,  # 2000 mmol/m3
    )
