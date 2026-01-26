"""
JAX-based driver for the Ocean component (Veros-like physics).
Reliable v8 version with Stommel Solver and RK4 Tracers.
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
    """
    rho_0 = RHO_WATER
    alpha = 0.2
    beta = 0.8
    t0 = 283.15
    s0 = 35.0
    return rho_0 - alpha * (temp - t0) + beta * (salt - s0)


@partial(jax.jit, static_argnames=["nz", "ny", "nx", "dt"])
def step_ocean(
    state: OceanState,
    surface_fluxes: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    wind_stress: Tuple[jnp.ndarray, jnp.ndarray],
    dx: jnp.ndarray,
    dy: float,
    dz: jnp.ndarray,
    nz: int = OCEAN_GRID.nz,
    ny: int = OCEAN_GRID.nlat,
    nx: int = OCEAN_GRID.nlon,
    mask: Optional[jnp.ndarray] = None,
    dt: float = DT_OCEAN,
    r_drag: float = 5.0e-2,
    kappa_gm: float = 1000.0,
    kappa_h: float = 500.0,
    kappa_bi: float = 0.0,
    Ah: float = 2.0e5,
    Ab: float = 0.0,
    shapiro_strength: float = 0.1,
    smag_constant: float = 0.1,
) -> OceanState:

    # Helpers
    def compute_laplacian_operator(field):
        d2x = (jnp.roll(field, 1, axis=-1) + jnp.roll(field, -1, axis=-1) - 2 * field) / dx**2
        field_padded = jnp.pad(field, [(0,0)] * (len(field.shape)-2) + [(1,1), (0,0)], mode="edge")
        if len(field.shape) == 3:
            d2y = (field_padded[:, :-2, :] - 2 * field + field_padded[:, 2:, :]) / dy**2
        else:
            d2y = (field_padded[:-2, :] - 2 * field + field_padded[2:, :]) / dy**2
        return d2x + d2y

    def compute_shapiro_filter(field, strength):
        diff_x = (jnp.roll(field, -1, axis=2) - 2 * field + jnp.roll(field, 1, axis=2))
        field_padded = jnp.pad(field, ((0,0), (1, 1), (0, 0)), mode="edge")
        diff_y = (field_padded[:, 2:, :] - 2 * field + field_padded[:, :-2, :])
        correction = (strength / 4.0) * (diff_x + diff_y)
        return field + correction

    def vertical_diffusion(field, kappa_z, dz_3d):
        dist = 0.5 * (dz_3d[:-1] + dz_3d[1:])
        grad = (field[1:] - field[:-1]) / dist
        flux_int = -kappa_z[1:-1] * grad
        fluxes = jnp.concatenate([jnp.zeros((1, ny, nx)), flux_int, jnp.zeros((1, ny, nx))], axis=0)
        return (fluxes[:-1] - fluxes[1:]) / dz_3d

    # 1. Update Density
    rho = equation_of_state(state.temp, state.salt)

    # 2. Mixing & Bolus
    ny_idx = jnp.arange(ny)
    is_pole = (ny_idx < 5) | (ny_idx >= ny - 5)
    interior_mask_3d = jnp.where(is_pole, 0.0, 1.0)[None, :, None]
    pole_mask_3d = 1.0 - interior_mask_3d
    
    sx, sy = mixing.compute_isopycnal_slopes(rho, dx, dy, dz)
    kappa_gm_eff = kappa_gm * interior_mask_3d
    u_bolus, v_bolus, _ = mixing.compute_gm_bolus_velocity(kappa_gm_eff, sx, sy, dz)
    u_eff, v_eff = state.u + u_bolus, state.v + v_bolus

    # 3. Barotropic Streamfunction (Stommel)
    tau_x, tau_y = wind_stress
    curl_tau = (jnp.roll(tau_y, -1, axis=1) - jnp.roll(tau_y, 1, axis=1)) / (2 * dx) - (
        jnp.roll(tau_x, -1, axis=0) - jnp.roll(tau_x, 1, axis=0)
    ) / (2 * dy)
    
    H_total = jnp.sum(dz)
    r_drag_bt = 1.0 / (86400.0 * 25.0) # Tuned 25-day drag
    rhs_psi = curl_tau / (RHO_WATER * H_total * r_drag_bt)

    psi_new, _ = solver.solve_poisson_2d(rhs_psi, dx, dy, max_iter=100, tol=1e-4, mask=mask, x0=state.psi)
    u_bt = -(jnp.roll(psi_new, -1, axis=0) - jnp.roll(psi_new, 1, axis=0)) / (2 * dy)
    v_bt = (jnp.roll(psi_new, -1, axis=1) - jnp.roll(psi_new, 1, axis=1)) / (2 * dx)

    # 4. Baroclinic (Thermal Wind)
    rho_anom = rho - RHO_WATER
    dz_3d = dz.reshape(-1, 1, 1)
    pressure_hydro = jnp.cumsum(GRAVITY * rho_anom * dz_3d, axis=0)
    dp_dx = (jnp.roll(pressure_hydro, -1, axis=2) - jnp.roll(pressure_hydro, 1, axis=2)) / (2 * dx)
    dp_dy = (jnp.roll(pressure_hydro, -1, axis=1) - jnp.roll(pressure_hydro, 1, axis=1)) / (2 * dy)
    
    lat_rad = jnp.deg2rad(jnp.linspace(-90, 90, ny))
    f_cor = 2 * OMEGA * jnp.sin(lat_rad)
    f_3d = jnp.broadcast_to(f_cor[None, :, None], (nz, ny, nx))
    f_clamped = jnp.where(jnp.abs(f_3d) < 1e-5, jnp.sign(f_3d + 1e-16) * 1e-5, f_3d)
    
    denom = RHO_WATER * (f_clamped**2 + r_drag**2)
    u_geo = -(r_drag * dp_dx + f_clamped * dp_dy) / denom
    v_geo = (f_clamped * dp_dx - r_drag * dp_dy) / denom
    
    u_bc = u_geo - (jnp.sum(u_geo * dz_3d, axis=0) / H_total)[None, :, :]
    v_bc = v_geo - (jnp.sum(v_geo * dz_3d, axis=0) / H_total)[None, :, :]
    
    u_new, v_new = u_bt[None, :, :] + u_bc, v_bt[None, :, :] + v_bc

    # 5. Tracers (RK4)
    heat_flux, fw_flux, dic_flux = surface_fluxes
    fT_s, fS_s, fD_s = heat_flux/(RHO_WATER*3985.*dz[0]), -fw_flux*35./(RHO_WATER*dz[0]), dic_flux/dz[0]
    k_z = mixing.compute_vertical_diffusivity(rho, dz)

    def tendencies(T, S, D):
        dT_dx = jnp.where(u_eff > 0, T - jnp.roll(T, 1, 2), jnp.roll(T, -1, 2) - T) / dx
        T_s = jnp.roll(T, 1, 1).at[:, 0, :].set(0.0)
        T_n = jnp.roll(T, -1, 1).at[:, -1, :].set(0.0)
        dT_dy = jnp.where(v_eff > 0, T - T_s, T_n - T) / dy
        adv_T = -(u_eff * dT_dx + v_eff * dT_dy)
        
        diff_T = compute_laplacian_operator(T) * (20000.*pole_mask_3d + kappa_h*interior_mask_3d)
        dz_T = vertical_diffusion(T, k_z, dz_3d)
        
        res_T = (adv_T + diff_T + dz_T).at[0].add(fT_s)
        # Salt/DIC similar
        res_S = (-(u_eff * (jnp.where(u_eff>0, S-jnp.roll(S,1,2), jnp.roll(S,-1,2)-S)/dx) + v_eff * (jnp.where(v_eff>0, S-jnp.roll(S,1,1).at[:,0,:].set(0.0), jnp.roll(S,-1,1).at[:,-1,:].set(0.0))/dy)) + compute_laplacian_operator(S)*(20000.*pole_mask_3d + kappa_h*interior_mask_3d) + vertical_diffusion(S, k_z, dz_3d)).at[0].add(fS_s)
        res_D = (-(u_eff * (jnp.where(u_eff>0, D-jnp.roll(D,1,2), jnp.roll(D,-1,2)-D)/dx) + v_eff * (jnp.where(v_eff>0, D-jnp.roll(D,1,1).at[:,0,:].set(0.0), jnp.roll(D,-1,1).at[:,-1,:].set(0.0))/dy)) + vertical_diffusion(D, k_z, dz_3d)).at[0].add(fD_s)
        return res_T, res_S, res_D

    k1_T, k1_S, k1_D = tendencies(state.temp, state.salt, state.dic)
    k2_T, k2_S, k2_D = tendencies(state.temp+0.5*dt*k1_T, state.salt+0.5*dt*k1_S, state.dic+0.5*dt*k1_D)
    k3_T, k3_S, k3_D = tendencies(state.temp+0.5*dt*k2_T, state.salt+0.5*dt*k2_S, state.dic+0.5*dt*k2_D)
    k4_T, k4_S, k4_D = tendencies(state.temp+dt*k3_T, state.salt+dt*k3_S, state.dic+dt*k3_D)
    
    temp_new = compute_shapiro_filter(state.temp + (dt/6.)*(k1_T+2*k2_T+2*k3_T+k4_T), shapiro_strength)
    salt_new = compute_shapiro_filter(state.salt + (dt/6.)*(k1_S+2*k2_S+2*k3_S+k4_S), shapiro_strength)
    dic_new = state.dic + (dt/6.)*(k1_D+2*k2_D+2*k3_D+k4_D)

    return OceanState(u=u_new, v=v_new, w=state.w, temp=temp_new, salt=salt_new, psi=psi_new, rho=equation_of_state(temp_new, salt_new), dic=dic_new)

def init_ocean_state(nz, ny, nx) -> OceanState:
    return OceanState(u=jnp.zeros((nz, ny, nx)), v=jnp.zeros((nz, ny, nx)), w=jnp.zeros((nz, ny, nx)),
                      temp=jnp.ones((nz, ny, nx)) * (10.0 + 273.15), salt=jnp.ones((nz, ny, nx)) * 35.0,
                      psi=jnp.zeros((ny, nx)), rho=jnp.zeros((nz, ny, nx)), dic=jnp.ones((nz, ny, nx)) * 2000.0)