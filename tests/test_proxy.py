"""
Tests for the Proxy Layer.
"""

import pytest
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm.proxy import forward_ops, sensor, loss

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


class TestSensor:
    """Test sensor masking and sampling."""
    
    def test_sample_nearest(self):
        """Test nearest neighbor sampling."""
        # Grid: 0, 10, 20
        lat_grid = jnp.array([0.0, 10.0, 20.0])
        lon_grid = jnp.array([0.0, 10.0, 20.0])
        
        # Field: Values equal to lat index
        field = jnp.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ])
        
        # Target: 11.0 (should map to index 1 -> value 1.0)
        target_lats = jnp.array([11.0])
        target_lons = jnp.array([5.0])
        
        val = sensor.sample_at_coords(
            field, lat_grid, lon_grid, target_lats, target_lons
        )
        
        assert jnp.isclose(val[0], 1.0)


class TestLoss:
    """Test loss functions."""
    
    def test_mse_masked(self):
        """Test masked MSE."""
        model = jnp.array([1.0, 2.0, 3.0])
        obs = jnp.array([1.0, 5.0, 3.0])
        mask = jnp.array([True, True, False]) # Ignore last element
        
        # Error at index 1: (2-5)^2 = 9
        # Index 0: 0
        # Index 2: Ignored
        # Mean: 9 / 2 = 4.5
        
        l = loss.mse_loss(model, obs, mask)
        assert jnp.isclose(l, 4.5)

if __name__ == "__main__":
    pass
