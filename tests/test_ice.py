"""
Tests for the Sea Ice component.
"""

import sys
from pathlib import Path

import jax  # noqa: F401
import jax.numpy as jnp
import pytest  # noqa: F401

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm.ice import driver, thermodynamics  # noqa: E402


class TestIcePhysics:
    """Test ice thermodynamics."""

    def test_albedo(self):
        """Test albedo transition."""
        # Cold thick ice -> High albedo
        alb_cold = thermodynamics.compute_albedo(jnp.array(-10.0), jnp.array(2.0))
        assert jnp.isclose(alb_cold, thermodynamics.ALBEDO_ICE, atol=0.01)

        # Melting ice -> Lower albedo
        alb_melt = thermodynamics.compute_albedo(jnp.array(0.0), jnp.array(2.0))
        assert alb_melt < alb_cold

        # Thin ice -> Ocean albedo
        alb_thin = thermodynamics.compute_albedo(jnp.array(-10.0), jnp.array(0.0))
        assert jnp.isclose(alb_thin, thermodynamics.ALBEDO_OCEAN, atol=0.01)

    def test_surface_solver(self):
        """Test surface temperature solver."""
        # Cold air, no sun -> Surface should be cold
        t_air = -20.0
        sw = 0.0
        lw = 200.0  # Low LW
        h = 1.0

        ts, _ = thermodynamics.solve_surface_temp(
            jnp.array(t_air), jnp.array(sw), jnp.array(lw), jnp.array(h)
        )

        # Surface should be between air and ocean (-1.8)
        assert ts > t_air
        assert ts < -1.8

        # Warm air, sun -> Surface should be capped at 0
        t_air = 10.0
        sw = 300.0
        lw = 300.0

        ts_warm, _ = thermodynamics.solve_surface_temp(
            jnp.array(t_air), jnp.array(sw), jnp.array(lw), jnp.array(h)
        )

        assert jnp.isclose(ts_warm, 0.0, atol=1e-4)


class TestIceDriver:
    """Test ice driver."""

    def test_freezing(self):
        """Test frazil ice formation."""
        ny, nx = 10, 10
        state = driver.init_ice_state(ny, nx)

        # Supercooled ocean
        ocean_temp = jnp.ones((ny, nx)) * -3.0  # Below -1.8

        # Run step
        new_state, _ = driver.step_ice(
            state,
            t_air=jnp.ones((ny, nx)) * -10.0,
            sw_down=jnp.zeros((ny, nx)),
            lw_down=jnp.zeros((ny, nx)),
            ocean_temp=ocean_temp,
            ny=ny,
            nx=nx,
        )

        # Should form ice
        assert jnp.all(new_state.thickness > 0.0)
        assert jnp.all(new_state.concentration > 0.0)

    def test_melting(self):
        """Test ice melting."""
        ny, nx = 10, 10
        # Start with ice
        state = driver.IceState(
            thickness=jnp.ones((ny, nx)) * 1.0,
            concentration=jnp.ones((ny, nx)),
            surface_temp=jnp.ones((ny, nx)) * -5.0,
        )

        # Warm conditions
        new_state, _ = driver.step_ice(
            state,
            t_air=jnp.ones((ny, nx)) * 10.0,
            sw_down=jnp.ones((ny, nx)) * 300.0,
            lw_down=jnp.ones((ny, nx)) * 300.0,
            ocean_temp=jnp.ones((ny, nx)) * -1.8,
            ny=ny,
            nx=nx,
        )

        # Should melt (thickness decrease)
        assert jnp.all(new_state.thickness < 1.0)


if __name__ == "__main__":
    pass
