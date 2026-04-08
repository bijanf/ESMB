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


def create_atlantic_mask(ny: int = OCEAN_GRID.nlat, nx: int = OCEAN_GRID.nlon) -> jnp.ndarray:
    """
    Create a 2D boolean mask for the Atlantic basin.

    Atlantic defined as: longitude 280-360E + 0-20E, latitude 34S-80N.
    Longitude indices assume uniform grid from 0 to 360.

    Returns:
        mask: (ny, nx) boolean array, True inside the Atlantic.
    """
    lon = jnp.linspace(0, 360, nx, endpoint=False)
    lat = jnp.linspace(-90, 90, ny)

    # Longitude: 280-360 OR 0-20 (wrapping)
    lon_mask = (lon >= 280.0) | (lon <= 20.0)  # (nx,)

    # Latitude: 34S to 80N
    lat_mask = (lat >= -34.0) & (lat <= 80.0)  # (ny,)

    # Outer product -> (ny, nx)
    return lat_mask[:, None] & lon_mask[None, :]


def compute_amoc(state: OceanState, atlantic_mask: jnp.ndarray = None,
                 dz: float = None) -> dict:
    """
    Compute Atlantic Meridional Overturning Circulation (AMOC).

    Returns a dict with:
        streamfunction: (nz, ny) Atlantic overturning in Sv
        upper_cell_26N: Upper cell strength at 26.5N [Sv]
        lower_cell_26N: Lower cell strength at 26.5N [Sv] (AABW)
    """
    nz, ny, nx = state.v.shape

    if atlantic_mask is None:
        atlantic_mask = create_atlantic_mask(ny, nx)

    if dz is None:
        dz = 5000.0 / nz

    # Latitude-dependent dx
    lat = jnp.linspace(-90, 90, ny)
    lat_rad = jnp.deg2rad(lat)
    dx = 2 * jnp.pi * EARTH_RADIUS * jnp.cos(lat_rad) / nx  # (ny,)
    dx_2d = jnp.broadcast_to(dx[:, None], (ny, nx))  # (ny, nx)

    # Mask v to Atlantic only
    mask_3d = jnp.broadcast_to(atlantic_mask[None, :, :], (nz, ny, nx))
    v_atlantic = jnp.where(mask_3d, state.v, 0.0)

    # Transport: v * dx  (m^2/s per grid cell)
    v_transport = v_atlantic * dx_2d[None, :, :]

    # Sum over longitude -> (nz, ny)
    v_zonal = jnp.sum(v_transport, axis=2)

    # Integrate from top down: Psi(k) = -Sum_{i=0}^{k} v_i * dz
    amoc_sv = -jnp.cumsum(v_zonal * dz, axis=0) / 1.0e6

    # Extract metrics at 26.5N (RAPID array latitude)
    lat_26n_idx = jnp.argmin(jnp.abs(lat - 26.5))
    profile_26n = amoc_sv[:, lat_26n_idx]

    upper_cell = jnp.max(profile_26n)  # Max positive (northward upper)
    lower_cell = jnp.min(profile_26n)  # Min negative (AABW)

    return {
        "streamfunction": amoc_sv,
        "upper_cell_26N": upper_cell,
        "lower_cell_26N": lower_cell,
    }


def compute_amoc_diagnostics(state: OceanState, atlantic_mask: jnp.ndarray = None) -> dict:
    """
    Compute a dict of scalar AMOC-related diagnostics for logging.

    Returns dict with:
        amoc_upper_26N: Upper cell at 26.5N [Sv]
        amoc_lower_26N: Lower cell at 26.5N [Sv]
        amoc_max: Global max of Atlantic overturning [Sv]
        north_atlantic_sst: Mean SST in N. Atlantic (40-60N, Atlantic lon) [K]
        global_mean_salinity: Volume-mean salinity [psu]
    """
    nz, ny, nx = state.v.shape

    if atlantic_mask is None:
        atlantic_mask = create_atlantic_mask(ny, nx)

    amoc = compute_amoc(state, atlantic_mask)

    # N. Atlantic SST (40-60N within Atlantic mask)
    lat = jnp.linspace(-90, 90, ny)
    na_lat_mask = (lat >= 40.0) & (lat <= 60.0)
    na_mask = na_lat_mask[:, None] & atlantic_mask
    na_sst = jnp.where(na_mask, state.temp[0], 0.0)
    na_count = jnp.sum(na_mask.astype(jnp.float32))
    na_sst_mean = jnp.sum(na_sst) / jnp.maximum(na_count, 1.0)

    # Global mean salinity
    global_mean_salt = jnp.mean(state.salt)

    return {
        "amoc_upper_26N": float(amoc["upper_cell_26N"]),
        "amoc_lower_26N": float(amoc["lower_cell_26N"]),
        "amoc_max": float(jnp.max(amoc["streamfunction"])),
        "north_atlantic_sst": float(na_sst_mean),
        "global_mean_salinity": float(global_mean_salt),
    }
