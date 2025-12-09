"""
Tests for the Atmosphere component.
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest  # noqa: F401

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm.atmos import dynamics, physics, qtcm, spectral  # noqa: E402


class TestAtmosPhysics:
    """Test atmospheric physics."""

    def test_precipitation(self):
        """Test softplus precipitation."""
        q = jnp.array([0.01, 0.02])
        q_sat = jnp.array([0.02, 0.02])

        # Case 1: q < q_sat (0.01 < 0.02) -> Should be near zero
        # Case 2: q = q_sat (0.02 = 0.02) -> Should be small positive (softplus)

        precip, heating = physics.compute_precipitation(q, q_sat)

        assert precip[0] < precip[1]
        assert jnp.all(precip >= 0)
        assert jnp.all(heating >= 0)

    def test_anthropogenic_forcing(self):
        """Test CO2 forcing."""
        temp = jnp.array([280.0])

        # Reference CO2
        h_ref = physics.compute_radiative_forcing(temp, 280.0)

        # Doubled CO2
        h_dbl = physics.compute_radiative_forcing(temp, 560.0)

        # Doubling CO2 should increase heating (or reduce cooling)
        assert h_dbl > h_ref

        # Check gradient
        grad_co2 = jax.grad(
            lambda c: jnp.sum(physics.compute_radiative_forcing(temp, c))
        )
        g = grad_co2(300.0)

        assert jnp.isfinite(g)
        assert g > 0  # More CO2 -> More heating


class TestSpectral:
    """Test spectral operations."""

    def test_gradients(self):
        """Test gradient computation."""
        nx, ny = 32, 16
        dx, dy = 1000.0, 1000.0

        # Linear field: f = x + y
        x = jnp.linspace(0, nx * dx, nx, endpoint=False)
        y = jnp.linspace(0, ny * dy, ny, endpoint=False)
        X, Y = jnp.meshgrid(x, y)
        f = X + Y

        df_dx, df_dy = spectral.compute_gradients(f, dx, dy)

        # Should be approx 1.0
        # Interior points
        assert jnp.allclose(df_dx[:, 2:-2], 1.0, rtol=1e-2)
        assert jnp.allclose(df_dy[2:-2, :], 1.0, rtol=1e-2)

    def test_laplacian_inverse(self):
        """Test Laplacian and its inverse."""
        nx, ny = 32, 32
        dx, dy = 1.0, 1.0

        # Eigenfunction of Laplacian: sin(kx)sin(ly)
        x = jnp.linspace(0, 2 * jnp.pi, nx, endpoint=False)
        y = jnp.linspace(0, 2 * jnp.pi, ny, endpoint=False)
        X, Y = jnp.meshgrid(x, y)

        f = jnp.sin(X) * jnp.sin(Y)
        # Laplacian should be -2 * f

        lap = spectral.compute_laplacian(f, dx, dy)

        # Inverse
        f_rec = spectral.inverse_laplacian(lap, dx, dy)

        # Check correlation or error
        # Note: Inverse is unique up to constant (mean)
        f_rec = f_rec - jnp.mean(f_rec)
        f = f - jnp.mean(f)

        err = jnp.linalg.norm(f - f_rec) / jnp.linalg.norm(f)
        # This might be loose due to FD approx vs exact spectral
        assert err < 0.1


class TestAtmosStepper:
    """Test main stepper."""

    def test_step(self):
        """Run a single step."""
        ny, nx = 10, 20
        state = qtcm.init_atmos_state(ny, nx)
        sst = jnp.ones((ny, nx)) * 300.0
        co2 = 400.0
        dx, dy = 100e3, 100e3

        new_state, fluxes = qtcm.step_atmos(state, sst, co2, dx, dy, ny, nx)

        assert new_state.temp.shape == (ny, nx)
        assert len(fluxes) == 2


if __name__ == "__main__":
    pass
