"""
Tests for the Coupler component.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm.config import ATMOS_GRID, OCEAN_GRID
from chronos_esm.coupler import regrid, state


class TestCoupler:
    """Test coupler functionality."""

    def test_regrid_identity(self):
        """Test regridding on identical grids (identity)."""
        regridder = regrid.Regridder()

        # Random field
        key = jax.random.PRNGKey(0)
        field = jax.random.uniform(key, (ATMOS_GRID.nlat, ATMOS_GRID.nlon))

        # Atmos -> Ocean
        field_ocn = regridder.atmos_to_ocean(field)

        # Should be identical for T63->T63
        assert jnp.allclose(field, field_ocn)

        # Ocean -> Atmos
        field_atm = regridder.ocean_to_atmos(field_ocn)
        assert jnp.allclose(field, field_atm)

    def test_conservation(self):
        """Test global conservation during regridding."""
        regridder = regrid.Regridder()

        # Field with mass
        field = jnp.ones((ATMOS_GRID.nlat, ATMOS_GRID.nlon))

        # Total mass (sum)
        mass_src = jnp.sum(field)

        # Regrid
        field_dest = regridder.atmos_to_ocean(field)
        mass_dest = jnp.sum(field_dest)

        # Should be conserved (within float precision)
        assert jnp.isclose(mass_src, mass_dest, rtol=1e-6)

    def test_differentiability(self):
        """Verify differentiability through regridding."""
        regridder = regrid.Regridder()

        def loss_fn(x):
            # Regrid and compute norm
            y = regridder.atmos_to_ocean(x)
            return jnp.sum(y**2)

        x = jnp.ones((ATMOS_GRID.nlat, ATMOS_GRID.nlon))
        grad = jax.grad(loss_fn)(x)

        assert jnp.all(jnp.isfinite(grad))
        assert jnp.linalg.norm(grad) > 0


if __name__ == "__main__":
    pass
