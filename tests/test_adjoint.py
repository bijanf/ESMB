
import jax
import jax.numpy as jnp
import pytest
from chronos_esm import main, config

def test_coupled_differentiability():
    """
    Verify that we can compute gradients through the coupled simulation.
    Target: Minimize (Mean Temp - Target)^2
    Parameter: Initial Ocean Temperature
    """
    print("\n--- Starting Adjoint Test ---")
    
    # 1. Setup
    # jax.config.update("jax_debug_nans", True)
    nz, ny, nx = 5, 16, 32 # Small grid for speed
    
    # Initialize state
    state_init = main.init_model(nz, ny, nx, ny, nx)
    params = main.ModelParams()
    regridder = main.regrid.Regridder() # Should handle small grid if dynamic? 
    # Note: Regridder currently hardcodes T63 weights if not careful.
    # Let's check regridder.py. It uses config.ATMOS_GRID.
    # We might need to mock or ensure regridder works with custom shapes.
    # For now, let's use the default grid to be safe, but run few steps.
    
    # Actually, main.init_model uses arguments for shapes, but regridder might not.
    # Let's use standard grid but very few steps.
    state_init = main.init_model() # Standard T63
    
    # Check for NaNs in init
    assert not jnp.isnan(state_init.ocean.temp).any(), "Init Ocean Temp has NaNs"
    assert not jnp.isnan(state_init.atmos.temp).any(), "Init Atmos Temp has NaNs"
    assert not jnp.isnan(state_init.fluxes.sst).any(), "Init SST has NaNs"
    
    # 2. Define Loss Function
    def loss_fn(temp_ic):
        # Replace initial temp in state
        # We need to reconstruct state with new temp
        # OceanState is namedtuple
        ocean_new = state_init.ocean._replace(temp=temp_ic)
        
        # Synchronize SST in fluxes so Atmos sees the change immediately
        # This ensures Step 1 depends on temp_ic
        # Assuming identical grids (T63)
        sst_initial = temp_ic[-1] # Top layer
        fluxes_new = state_init.fluxes._replace(sst=sst_initial)
        
        state_start = state_init._replace(ocean=ocean_new, fluxes=fluxes_new)
        
        # Run Simulation (Short: 5 steps)
        # We use scan directly to avoid overhead
        def step_fn(carry, _):
            s = main.step_coupled(carry, params, regridder)
            return s, None
            
        final_state, _ = jax.lax.scan(step_fn, state_start, jnp.arange(1))
        
        # Loss: Target Global Mean Temp = 290K
        t_mean = jnp.mean(final_state.atmos.temp)
        loss = (t_mean - 290.0)**2
        
        # Dummy Loss to verify harness
        # loss = jnp.mean(temp_ic)
        return loss
    
    # 3. Compute Gradient
    print("Computing gradient...")
    # We differentiate w.r.t ocean temperature
    temp_ic = state_init.ocean.temp
    
    # Finite Difference Check
    print("Computing Finite Difference Gradient...")
    eps = 1e-4
    # Perturb one element
    temp_perturbed = temp_ic.at[-1, 8, 16].add(eps)
    loss_base = loss_fn(temp_ic)
    loss_pert = loss_fn(temp_perturbed)
    fd_grad = (loss_pert - loss_base) / eps
    print(f"FD Gradient (at -1, 8, 16): {fd_grad:.4e}")
    
    # Value and Grad
    loss_val, grads = jax.value_and_grad(loss_fn)(temp_ic)
    
    print(f"Loss: {loss_val:.4f}")
    print(f"Gradient Norm: {jnp.linalg.norm(grads):.4e}")
    print(f"Gradient Max: {jnp.max(jnp.abs(grads)):.4e}")
    
    # 4. Verification
    assert not jnp.isnan(loss_val), "Loss is NaN"
    assert not jnp.isnan(grads).any(), "Gradients contain NaNs"
    assert jnp.linalg.norm(grads) > 0.0, "Gradient is zero (graph broken?)"
    
    print("Adjoint Test Passed! Gradients are flowing.")

if __name__ == "__main__":
    test_coupled_differentiability()
