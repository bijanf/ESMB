"""
Dense matrix regridding for Chronos-ESM.

Handles conservative remapping between Atmosphere and Ocean grids.
"""

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from chronos_esm.config import ATMOS_GRID, OCEAN_GRID

# For T63, both grids are 192x96.
# If they are identical, the weight matrix is Identity.
# If they are different (e.g. Arakawa B vs C, or different resolutions),
# we would load a pre-computed weight matrix.

# For this implementation, we assume collocated grids for simplicity,
# but structure the code to use matrix multiplication as required.


def create_identity_weights(n_src: int, n_dest: int) -> jnp.ndarray:
    """Create identity weight matrix (for collocated grids)."""
    if n_src != n_dest:
        # Simple linear interpolation or nearest neighbor could be generated here
        # But for now, we assume matching sizes or error
        raise ValueError(
            f"Grid sizes must match for identity weights: {n_src} != {n_dest}"
        )
    return jnp.eye(n_src)


@partial(jax.jit, static_argnames=["n_src", "n_dest"])
def regrid_field(
    field: jnp.ndarray, weights: Optional[jnp.ndarray], n_src: int, n_dest: int
) -> jnp.ndarray:
    """
    Regrid a field using sparse or dense matrix multiplication.

    Dest = W @ Src

    Args:
        field: Source field (..., n_src) - flattened spatial dim
        weights: Weight matrix (n_dest, n_src). If None, assume Identity/Pass-through.

    Returns:
        Regridded field (..., n_dest)
    """
    # Optimization: If weights is None, assume identity/pass-through
    # This avoids expensive matrix multiplication for collocated grids
    if weights is None:
        # Just reshape if needed
        # For T63 (96, 192), size = 18432
        if n_dest == 18432:
            # Ensure last dims match target shape
            if field.shape[-2:] == (96, 192):
                return field
            else:
                return field.reshape(*field.shape[:-1], 96, 192)
        return field

    # Flatten last dimensions if needed
    # Assuming input is (..., ny, nx) -> (..., n_points)

    # If field is 2D (ny, nx), flatten it
    original_shape = field.shape
    if field.shape[-1] * field.shape[-2] == n_src:
        field_flat = field.reshape(*field.shape[:-2], -1)
    elif field.shape[-1] == n_src:
        field_flat = field
    else:
        # Fallback or error
        field_flat = field.reshape(-1, n_src)

    # Matrix multiplication: (..., n_src) @ (n_src, n_dest) -> (..., n_dest)
    # Wait, W is (n_dest, n_src). So we want W @ field_flat.T?
    # Or field_flat @ W.T

    result_flat = jnp.dot(field_flat, weights.T)

    # Reshape back to 2D grid if possible
    # We need to know the dest grid shape.
    # For T63 (96, 192), n_dest = 18432.
    if n_dest == 18432:
        return result_flat.reshape(*result_flat.shape[:-1], 96, 192)

    return result_flat


class Regridder:
    """
    Manages regridding weights and operations.
    """

    def __init__(self):
        # Initialize weights
        # In a real app, load from NetCDF
        self.n_atmos = ATMOS_GRID.size
        self.n_ocean = OCEAN_GRID.nlat * OCEAN_GRID.nlon  # Surface only usually

        # For T63 collocated:
        # Optimization: Use None for identity to skip matmul
        if self.n_atmos == self.n_ocean:
            self.w_atmos_to_ocean = None
            self.w_ocean_to_atmos = None
        else:
            self.w_atmos_to_ocean = create_identity_weights(self.n_atmos, self.n_ocean)
            self.w_ocean_to_atmos = create_identity_weights(self.n_ocean, self.n_atmos)

    def atmos_to_ocean(self, field: jnp.ndarray) -> jnp.ndarray:
        return regrid_field(field, self.w_atmos_to_ocean, self.n_atmos, self.n_ocean)

    def ocean_to_atmos(self, field: jnp.ndarray) -> jnp.ndarray:
        return regrid_field(field, self.w_ocean_to_atmos, self.n_ocean, self.n_atmos)
