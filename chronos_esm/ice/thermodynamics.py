"""
Thermodynamic Sea Ice Model (Semtner 0-layer).

Implements:
1. Surface energy balance solver
2. Ice growth/melt physics
3. Albedo parameterization
"""

from typing import Callable, NamedTuple, Tuple  # noqa: F401

import jax
import jax.numpy as jnp

# Constants
RHO_ICE = 917.0  # Density of ice [kg/m^3]
CP_ICE = 2106.0  # Specific heat of ice [J/kg/K]
L_FUSION = 3.34e5  # Latent heat of fusion [J/kg]
K_ICE = 2.03  # Thermal conductivity of ice [W/m/K]
T_FREEZE = -1.8  # Freezing point of seawater [C] (approx)
SIGMA = 5.67e-8  # Stefan-Boltzmann constant
EMISSIVITY = 0.97  # Ice emissivity

# Albedo
ALBEDO_ICE = 0.7
ALBEDO_MELT = 0.5  # Melting ice / puddles
ALBEDO_OCEAN = 0.1  # Open water


def compute_albedo(surface_temp: jnp.ndarray, thickness: jnp.ndarray) -> jnp.ndarray:
    """
    Compute surface albedo with smooth transition for melting.

    Args:
        surface_temp: Surface temperature [C]
        thickness: Ice thickness [m]

    Returns:
        albedo: Surface albedo
    """
    # Smooth transition from frozen to melting albedo near 0C
    # Sigmoid transition width
    width = 0.5  # degrees C
    melt_fraction = jax.nn.sigmoid((surface_temp + 0.5) / width)

    ice_albedo = (1.0 - melt_fraction) * ALBEDO_ICE + melt_fraction * ALBEDO_MELT

    # Transition to ocean albedo for thin ice
    # h_ref = 0.5 m
    thin_ice_fraction = jax.nn.sigmoid((0.5 - thickness) / 0.1)

    return (1.0 - thin_ice_fraction) * ice_albedo + thin_ice_fraction * ALBEDO_OCEAN


def solve_surface_temp(
    t_air: jnp.ndarray,
    sw_down: jnp.ndarray,
    lw_down: jnp.ndarray,
    thickness: jnp.ndarray,
    t_ocean: jnp.ndarray = -1.8,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve for surface temperature Ts satisfying energy balance.

    Fluxes (positive down):
    F_net = (1-alpha)SW + LW_down - sigma*(Ts+273.15)^4 - Sensible - Latent + Cond

    Conductive flux from ocean: F_cond = k_ice * (T_freeze - Ts) / h

    Simplified balance:
    (1-alpha)SW + LW_down - sigma*Ts^4 + k*(Tf - Ts)/h = 0
    (Neglecting sensible/latent for this 0-layer approx, or linearizing them)

    We'll use a linearized Newton step or simple relaxation.
    Since Ts cannot exceed 0C (melting), we cap it.

    Args:
        t_air: Air temperature [C]
        sw_down: Shortwave flux [W/m^2]
        lw_down: Longwave flux [W/m^2]
        thickness: Ice thickness [m]
        t_ocean: Ocean temperature (freezing point) [C]

    Returns:
        ts: Surface temperature [C]
        net_flux: Net flux into ice [W/m^2]
    """
    # Initial guess: Air temperature
    ts = t_air

    # Effective conductivity (ice + snow could be added)
    # Avoid division by zero for thickness
    h_safe = jnp.maximum(thickness, 0.1)
    k_eff = K_ICE / h_safe

    # Linearized Stefan-Boltzmann: sigma*T^4 approx sigma*T0^4 + 4*sigma*T0^3 * (T-T0)
    # Let's just use the nonlinear equation and solve iteratively (Newton)
    # Or for differentiability, a fixed number of Newton steps (e.g. 3)

    def energy_balance(t_sfc):
        # Kelvin
        tk = t_sfc + 273.15

        # Albedo
        alpha = compute_albedo(t_sfc, thickness)

        # Net SW
        sw_net = (1.0 - alpha) * sw_down

        # LW Up
        lw_up = EMISSIVITY * SIGMA * tk**4

        # Conductive Flux (upward is negative in this convention? No, flux into surface)
        # Flux from bottom to top: k*(Tf - Ts)/h
        f_cond = k_eff * (t_ocean - t_sfc)

        # Sensible/Latent (Simplified linearization around T_air)
        # F_turb = C_bulk * (T_air - Ts)
        c_bulk = 10.0  # W/m^2/K approx
        f_turb = c_bulk * (t_air - t_sfc)

        # Net flux into surface
        return sw_net + lw_down - lw_up + f_cond + f_turb

    # Newton solver
    # dF/dTs = -4*sigma*T^3 - k/h - C_bulk
    def derivative(t_sfc):
        tk = t_sfc + 273.15
        df_dts = -4 * EMISSIVITY * SIGMA * tk**3 - k_eff - 10.0
        return df_dts

    # 3 iterations
    for _ in range(3):
        f_val = energy_balance(ts)
        df = derivative(ts)
        ts = ts - f_val / (df - 1e-5)

    # Cap at melting point
    ts = jnp.minimum(ts, 0.0)

    # Recompute balance at final Ts
    net_flux = energy_balance(ts)

    return ts, net_flux


def compute_growth(
    net_surface_flux: jnp.ndarray, thickness: jnp.ndarray, t_ocean: jnp.ndarray = -1.8
) -> jnp.ndarray:
    """
    Compute ice growth/melt rate.

    dh/dt = -F_net / (rho * L)

    If Ts < 0: F_net is the imbalance (should be 0). Growth is from conduction.
    Actually, in Semtner 0-layer:
    Growth at bottom: k*(Tf - Ts)/h - F_ocean
    Melt at top: If Ts=0, remaining energy melts ice.

    Args:
        net_surface_flux: Residual flux at surface [W/m^2] (positive heats ice)
        thickness: Ice thickness [m]

    Returns:
        dh_dt: Growth rate [m/s]
    """
    # Bottom growth (conduction - ocean heat)
    # Assume ocean heat flux is small or handled by coupler
    # Here we just do conductive growth
    # We need Ts for conduction. We should probably return it from solve_surface_temp
    # But let's approximate or assume net_surface_flux contains the melt energy

    # Wait, the standard way:
    # 1. Solve Ts. If Ts < 0, flux balance is 0. Growth is determined by conduction at bottom.
    # 2. If Ts = 0, flux balance is positive. This energy melts top.

    # Let's refine solve_surface_temp to return Ts.
    # We need to re-call it or pass it in.
    # For this function, let's assume we have the conductive flux available.

    # Simplified:
    # Net energy into ice column
    # E_net = Net_Surface + Net_Bottom
    # dh/dt = -E_net / (rho * L)

    # This function might be too simple. Let's handle growth in the driver
    # where we have all fluxes.
    pass
