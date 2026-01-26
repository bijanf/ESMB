"""
Diagnostics for the Ocean component.
"""

import jax
import jax.numpy as jnp

from chronos_esm.config import EARTH_RADIUS, OCEAN_GRID
from chronos_esm.ocean.veros_driver import OceanState


def compute_moc(state: OceanState, dx: float = None) -> jnp.ndarray:
    """
    Compute the Global Meridional Overturning Circulation (MOC).

    Psi(y, z) = - Integral_x Integral_z' v(x, y, z') dz' dx

    Args:
        state: OceanState
        dx: Grid spacing in x [m] (optional, can be approximated)

    Returns:
        moc: (nz, ny) Streamfunction in Sverdrups (10^6 m^3/s)
    """
    # v is (nz, ny, nx)
    # We integrate from bottom up or top down.
    # Standard: Integrate from bottom (z=-H) to z.
    # Psi(y, z) = Integral_{bottom}^{z} <v> dy dz ? No.

    # Definition:
    # Psi(y, z) = - Integral_{x_west}^{x_east} Integral_{z}^{surface} v(x, y, z') dz' dx
    # Or:
    # Psi(y, z) = Integral_{x_west}^{x_east} Integral_{bottom}^{z} v(x, y, z') dz' dx

    # Let's use the second form (bottom up).
    # Assuming dz is constant for now or we need to pass it.
    # In veros_driver, dz is passed to step_ocean. We should probably accept it here or assume it.
    # Let's assume constant dz for now or approximate.
    # Actually, let's use the dz from config if possible, or pass it.
    # For T63, we have NZ_OCEAN levels.

    # We'll sum over X first.
    # v_zonal_sum = sum(v * dx, axis=x)

    # If dx is not provided, approximate for T63
    if dx is None:
        # dx varies with latitude
        ny, nx = state.v.shape[1], state.v.shape[2]
        lat = jnp.linspace(-90, 90, ny)
        lat_rad = jnp.deg2rad(lat)
        dx = 2 * jnp.pi * EARTH_RADIUS * jnp.cos(lat_rad) / nx
        # Broadcast dx to (ny, nx)
        dx = jnp.broadcast_to(dx[:, None], (ny, nx))

    v_transport = state.v * dx[None, :, :]  # (nz, ny, nx) * (ny, nx) -> m^2/s

    # Sum over Longitude
    v_zonal = jnp.sum(v_transport, axis=2)  # (nz, ny) [m^2/s]

    # Integrate vertically (cumulative sum from bottom)
    # Assuming constant dz for simplicity if not passed.
    # In main.py: dz_ocn = jnp.ones(...) * 100.0
    dz = 5000.0 / 15.0 # Match model (333.33m)

    # Psi = Sum(v * dx * dz) from bottom
    # cumsum along axis 0 (z)
    # But usually z index 0 is top.
    # So we integrate from bottom (index -1) to top.

    # Let's integrate from top down:
    # Psi(z) = - Integral_{z}^{0} v dz
    # Psi(k) = - Sum_{i=0}^{k} v_i * dz

    moc = -jnp.cumsum(v_zonal * dz, axis=0)

    # Convert to Sverdrups (10^6 m^3/s)
    moc_sv = moc / 1.0e6

    return moc_sv


def compute_amoc_index(state: OceanState) -> float:
    """
    Compute a scalar AMOC index (Max overturning at ~30N).
    """
    moc = compute_moc(state)

    # Dynamic index calculation
    nlat = OCEAN_GRID.nlat
    lat_idx = int((30.0 - (-90.0)) / 180.0 * nlat)
    lat_idx = min(lat_idx, nlat - 1) # Safety clamp

    # Search for max in depth at this latitude
    # Usually AMOC is the max positive value in the upper cell
    amoc_profile = moc[:, lat_idx]
    amoc_index = jnp.max(amoc_profile)

    return amoc_index
