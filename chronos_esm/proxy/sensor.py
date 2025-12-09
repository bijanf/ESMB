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
        method: 'nearest' or 'bilinear' (bilinear not fully impl in this snippet)

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

    # Nearest neighbor
    y_idx_int = jnp.round(y_idx).astype(int)
    x_idx_int = jnp.round(x_idx).astype(int)

    # Clip to bounds
    ny, nx = field.shape
    y_idx_int = jnp.clip(y_idx_int, 0, ny - 1)
    x_idx_int = jnp.clip(x_idx_int, 0, nx - 1)  # Periodic handling needed for lon?

    return field[y_idx_int, x_idx_int]


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
