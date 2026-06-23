"""
Tests for the Proxy Layer.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest  # noqa: F401

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm.proxy import forward_ops, loss, sensor  # noqa: E402


class TestProxyOps:
    """Test proxy forward operators."""

    def test_bemis_equation(self):
        """Test Bemis d18O calculation."""
        # T = 16.5 - 4.8 * (d18Oc - d18Osw)
        # If T = 16.5 and d18Osw = 0, then d18Oc should be 0.
        sst = 16.5
        d18o = forward_ops.bemis_d18o(sst, 0.0)
        assert jnp.isclose(d18o, 0.0)

        # If T increases, d18Oc should decrease (inverse relationship)
        sst_warm = 25.0
        d18o_warm = forward_ops.bemis_d18o(sst_warm, 0.0)
        assert d18o_warm < 0.0

    def test_gradients(self):
        """Verify differentiability of forward op."""

        def op(t):
            return forward_ops.bemis_d18o(t, 0.0)

        grad = jax.grad(op)(16.5)
        # d(d18Oc)/dT = -1 / 4.8
        expected = -1.0 / 4.8
        assert jnp.isclose(grad, expected)

    def test_salinity_to_d18o_sw(self):
        """Linear seawater d18O--salinity relation (LeGrande & Schmidt 2006)."""
        salt = jnp.array([34.0, 35.0, 36.0])
        d18o_sw = forward_ops.salinity_to_d18o_sw(salt)
        # 0 at the reference salinity, slope 0.5 permil/psu, saltier -> heavier.
        assert jnp.isclose(d18o_sw[1], 0.0)
        assert jnp.isclose(d18o_sw[0], -0.5)
        assert jnp.isclose(d18o_sw[2], 0.5)
        assert d18o_sw[2] > d18o_sw[0]
        # differentiable in salinity with gradient == slope
        g = jax.grad(lambda s: forward_ops.salinity_to_d18o_sw(s).sum())(salt)
        assert jnp.allclose(g, 0.5)

    def test_full_chain_differentiable(self):
        """S -> d18O_sw -> d18O_calcite chain flows a gradient end to end."""

        def chain(sst_c, salt):
            return forward_ops.bemis_d18o(sst_c, forward_ops.salinity_to_d18o_sw(salt))

        # d(d18Oc)/dS = slope = 0.5 ; d(d18Oc)/dT = -1/4.8
        gs = jax.grad(chain, argnums=1)(20.0, 35.0)
        gt = jax.grad(chain, argnums=0)(20.0, 35.0)
        assert jnp.isclose(gs, 0.5)
        assert jnp.isclose(gt, -1.0 / 4.8)


class TestSensor:
    """Test sensor masking and sampling."""

    def test_sample_nearest(self):
        """Test nearest neighbor sampling."""
        # Grid: 0, 10, 20
        lat_grid = jnp.array([0.0, 10.0, 20.0])
        lon_grid = jnp.array([0.0, 10.0, 20.0])

        # Field: Values equal to lat index
        field = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

        # Target: 11.0 (should map to index 1 -> value 1.0)
        target_lats = jnp.array([11.0])
        target_lons = jnp.array([5.0])

        val = sensor.sample_at_coords(
            field, lat_grid, lon_grid, target_lats, target_lons
        )

        assert jnp.isclose(val[0], 1.0)

    def test_sample_bilinear(self):
        """Bilinear sampling interpolates between cells and stays differentiable."""
        lat_grid = jnp.array([0.0, 10.0, 20.0])
        lon_grid = jnp.array([0.0, 10.0, 20.0])
        # field linear in the latitude index: row r has value r
        field = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

        # Target at lat=5 (halfway between rows 0 and 1) -> 0.5
        val = sensor.sample_at_coords(
            field,
            lat_grid,
            lon_grid,
            jnp.array([5.0]),
            jnp.array([5.0]),
            method="bilinear",
        )
        assert jnp.isclose(val[0], 0.5)

        # gradient of the sampled value w.r.t. the field is nonzero (interpolating)
        def sampled(f):
            return sensor.sample_at_coords(
                f,
                lat_grid,
                lon_grid,
                jnp.array([5.0]),
                jnp.array([5.0]),
                method="bilinear",
            )[0]

        g = jax.grad(sampled)(field)
        assert float(jnp.sum(g)) > 0.0


class TestLoss:
    """Test loss functions."""

    def test_mse_masked(self):
        """Test masked MSE."""
        model = jnp.array([1.0, 2.0, 3.0])
        obs = jnp.array([1.0, 5.0, 3.0])
        mask = jnp.array([True, True, False])  # Ignore last element

        # Error at index 1: (2-5)^2 = 9
        # Index 0: 0
        # Index 2: Ignored
        # Mean: 9 / 2 = 4.5

        loss_val = loss.mse_loss(model, obs, mask)
        assert jnp.isclose(loss_val, 4.5)


if __name__ == "__main__":
    pass
