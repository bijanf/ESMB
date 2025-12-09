"""
Tests for the Ocean component.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm.ocean import mixing, veros_driver
from chronos_esm.config import OCEAN_GRID

class TestOceanMixing:
    """Test differentiable mixing schemes."""
    
    def test_slope_limiter(self):
        """Verify slope limiter is smooth and bounded."""
        slopes = jnp.linspace(0, 0.1, 100)
        limited = mixing.slope_limiter(slopes, max_slope=0.01)
        
        # Check boundedness
        assert jnp.all(jnp.abs(limited) <= 0.01 + 1e-6)
        
        # Check linearity for small slopes
        small_slope = 1e-5
        assert jnp.isclose(mixing.slope_limiter(small_slope), small_slope, rtol=1e-3)
        
        # Check differentiability
        grad_limiter = jax.grad(lambda s: jnp.sum(mixing.slope_limiter(s)))
        grads = grad_limiter(slopes)
        assert jnp.all(jnp.isfinite(grads))
        
    def test_isopycnal_slopes(self):
        """Test slope calculation on a tilted density field."""
        nz, ny, nx = 10, 20, 20
        dx, dy = 1000.0, 1000.0
        dz = jnp.ones(nz) * 100.0
        
        # Create a density field tilted in x: rho = z + 1e-3 * x
        z_coords = jnp.arange(nz) * 100.0
        x_coords = jnp.arange(nx) * 1000.0
        
        Z, _, X = jnp.meshgrid(z_coords, jnp.arange(ny), x_coords, indexing='ij')
        
        rho = 1025.0 + 0.1 * Z + 1e-4 * X
        
        sx, sy = mixing.compute_isopycnal_slopes(rho, dx, dy, dz)
        
        # Analytical slope:
        # drho/dx = 1e-4
        # drho/dz = 0.1
        # sx = - (1e-4) / (0.1) = -1e-3
        
        # Check interior values (boundaries might have padding effects)
        sx_interior = sx[1:-1, 1:-1, 1:-1]
        assert jnp.allclose(sx_interior, -1e-3, rtol=1e-2)
        assert jnp.allclose(sy, 0.0, atol=1e-8)


class TestOceanDriver:
    """Test the main ocean driver."""
    
    def test_step_execution(self):
        """Verify the ocean step runs without errors and shapes are correct."""
        nz, ny, nx = 5, 10, 10
        dx, dy = 100e3, 100e3 # 100km
        dz = jnp.ones(nz) * 100.0
        
        state = veros_driver.init_ocean_state(nz, ny, nx)
        
        # Dummy fluxes
        fluxes = (jnp.zeros((ny, nx)), jnp.zeros((ny, nx)))
        
        # Spatially varying wind stress to generate curl
        # tau_x = 0.1 * sin(2*pi*y/L)
        y_coords = jnp.linspace(0, 1, ny)
        tau_x = 0.1 * jnp.sin(2 * jnp.pi * y_coords[:, None] * jnp.ones((1, nx)))
        wind = (tau_x, jnp.zeros((ny, nx)))
        
        # Run one step
        new_state = veros_driver.step_ocean(
            state, fluxes, wind, dx, dy, dz, nz, ny, nx
        )
        
        # Check shapes
        assert new_state.u.shape == (nz, ny, nx)
        assert new_state.psi.shape == (ny, nx)
        
        # Check that psi is non-zero due to wind stress
        assert jnp.any(new_state.psi != 0)
        
    def test_differentiability(self):
        """Verify we can differentiate through the ocean step."""
        nz, ny, nx = 5, 10, 10
        dx, dy = 100e3, 100e3
        dz = jnp.ones(nz) * 100.0
        
        state = veros_driver.init_ocean_state(nz, ny, nx)
        fluxes = (jnp.zeros((ny, nx)), jnp.zeros((ny, nx)))
        
        def loss_fn(amp):
            # Varying wind stress: tau_x = amp * sin(y)
            y_coords = jnp.linspace(0, 1, ny)
            tau_x = amp * jnp.sin(2 * jnp.pi * y_coords[:, None] * jnp.ones((1, nx)))
            wind = (tau_x, jnp.zeros((ny, nx)))
            
            new_state = veros_driver.step_ocean(
                state, fluxes, wind, dx, dy, dz, nz, ny, nx
            )
            # Minimize kinetic energy
            return 0.5 * jnp.sum(new_state.u**2 + new_state.v**2)
            
        amplitude = 0.1
        
        # Compute gradient
        grad_amp = jax.grad(loss_fn)(amplitude)
        
        assert jnp.isfinite(grad_amp)
        # Gradient should be non-zero (wind affects velocity)
        assert jnp.abs(grad_amp) > 0.0

if __name__ == "__main__":
    # Manual run if needed
    pass
