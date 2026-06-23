"""
Sensor masking and interpolation for sparse proxy data.
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=["method"])
def sample_at_coords(
    field: jnp.ndarray,
    lat_grid: jnp.ndarray,
    lon_grid: jnp.ndarray,
    target_lats: jnp.ndarray,
    target_lons: jnp.ndarray,
    method: str = "nearest",
) -> jnp.ndarray:
    """
    Sample model field at specific target coordinates.

    Args:
        field: Model field (ny, nx)
        lat_grid: Grid latitudes (ny,) or (ny, nx)
        lon_grid: Grid longitudes (nx,) or (ny, nx)
        target_lats: Target latitudes (n_obs,)
        target_lons: Target longitudes (n_obs,)
        method: 'nearest' (default) or 'bilinear'. Bilinear interpolates the four
            surrounding cells (periodic in longitude, clamped in latitude) and is
            smooth in the target coordinates -- preferred when a gradient w.r.t. the
            sampled value or a sub-grid observation location is needed.

    Returns:
        sampled_values: Values at target locations (n_obs,)
    """
    # For T63, we have regular grid indices.
    # Map lat/lon to indices.

    # Assuming regular grid for simplicity of this prototype
    # lat index = (lat - lat_min) / dlat
    # lon index = (lon - lon_min) / dlon

    # If lat_grid is 1D:
    if lat_grid.ndim == 1:
        dlat = lat_grid[1] - lat_grid[0]
        lat0 = lat_grid[0]
        y_idx = (target_lats - lat0) / dlat
    else:
        # Handle curvilinear? For now assume 1D separable for indexing
        # Or use nearest neighbor search
        raise NotImplementedError("Only 1D lat/lon grids supported for simple indexing")

    if lon_grid.ndim == 1:
        dlon = lon_grid[1] - lon_grid[0]
        lon0 = lon_grid[0]
        x_idx = (target_lons - lon0) / dlon
    else:
        raise NotImplementedError("Only 1D lat/lon grids supported")

    ny, nx = field.shape

    if method == "nearest":
        y_idx_int = jnp.clip(jnp.round(y_idx).astype(int), 0, ny - 1)
        x_idx_int = jnp.clip(jnp.round(x_idx).astype(int), 0, nx - 1)
        return field[y_idx_int, x_idx_int]

    if method == "bilinear":
        # Four-corner weighted average; latitude clamped, longitude periodic.
        y0f = jnp.floor(y_idx)
        x0f = jnp.floor(x_idx)
        wy = y_idx - y0f
        wx = x_idx - x0f
        y0 = jnp.clip(y0f.astype(int), 0, ny - 1)
        y1 = jnp.clip((y0f + 1).astype(int), 0, ny - 1)
        x0 = jnp.mod(x0f.astype(int), nx)  # periodic longitude
        x1 = jnp.mod((x0f + 1).astype(int), nx)
        f00 = field[y0, x0]
        f01 = field[y0, x1]
        f10 = field[y1, x0]
        f11 = field[y1, x1]
        top = f00 * (1.0 - wx) + f01 * wx
        bot = f10 * (1.0 - wx) + f11 * wx
        return top * (1.0 - wy) + bot * wy

    raise ValueError(
        f"Unknown sampling method {method!r} (use 'nearest' or 'bilinear')"
    )


def create_mask(
    field_shape: Tuple[int, int], indices: Tuple[jnp.ndarray, jnp.ndarray]
) -> jnp.ndarray:
    """
    Create a boolean mask from indices.

    Args:
        field_shape: (ny, nx)
        indices: (y_indices, x_indices)

    Returns:
        mask: (ny, nx) boolean array
    """
    mask = jnp.zeros(field_shape, dtype=bool)
    mask = mask.at[indices].set(True)
    return mask
