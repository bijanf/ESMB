
import os
import sys
import jax
import jax.numpy as jnp
# import optax # Not available
import numpy as np
import time

sys.path.append(os.getcwd())

from chronos_esm import main, data
from chronos_esm.ocean import diagnostics

# Target AMOC (Sverdrups)
TARGET_AMOC = 17.0

def loss_fn(r_drag_param, state_init, params, regridder, steps=50):
    """
    Run model for 'steps' and compute loss based on AMOC deviation.
    r_drag_param is the parameter to optimize.
    """
    # Constrain r_drag to be positive (using softplus or similar if needed, here simple)
    r_drag = jax.nn.softplus(r_drag_param) # ensure positive
    
    def scan_fn(carry, _):
        state = carry
        state = main.step_coupled(state, params, regridder, r_drag=r_drag)
        return state, None
        
    final_state, _ = jax.lax.scan(scan_fn, state_init, None, length=steps)
    
    # Compute AMOC
    # We need to use valid dx, dz logic from diagnostics within JAX
    # But diagnostics.compute_amoc_index might use hardcoded grid? 
    # Let's import the function.
    
    amoc_val = diagnostics.compute_amoc_index(final_state.ocean)
    
    # Loss: Squared Error
    loss = (amoc_val - TARGET_AMOC)**2
    
    return loss, (amoc_val, r_drag)

def run_tuning():
    print("Starting JAX Auto-Tuning for r_drag...")
    
    # Initialize
    mask = jnp.array(data.load_bathymetry_mask())
    params = main.ModelParams(mask=mask)
    state = main.init_model()
    regridder = main.regrid.Regridder()
    
    # Initial Guess: r_drag = 0.05 (from manual tuning)
    initial_r = 0.05
    raw_r = np.log(np.exp(initial_r) - 1.0)
    
    # Optimizer (Manual SGD)
    learning_rate = 0.01
    raw_r_param = jnp.array(raw_r)
    
    # Tuning Loop
    n_epochs = 50
    steps_per_epoch = 100 # Short burst
    
    @jax.jit
    def update_step(raw_r, state_start):
        (loss, (amoc, current_r)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            raw_r, state_start, params, regridder, steps=steps_per_epoch
        )
        # Manual SGD Update
        # grads is a tuple/array matching raw_r structure (scalar)
        new_raw_r = raw_r - learning_rate * grads
        return new_raw_r, loss, amoc, current_r
    
    # We reset state each epoch or continue?
    # Ideally we want to find r that works for the EQUILIBRIUM.
    # But finding equilibrium inside grad loop is expensive.
    # Strategy: Run forward, taking gradients on short segments, "Online Learning".
    
    curr_state = state
    
    print(f"{'Epoch':<5} | {'Loss':<10} | {'AMOC (Sv)':<10} | {'r_drag':<10}")
    print("-" * 45)
    
    for epoch in range(n_epochs):
        start_t = time.time()
        raw_r_param, loss, amoc, current_r = update_step(raw_r_param, curr_state)
        
        # Advance state for next segment (using the NEW r)
        # Note: step_coupled is deterministic given state/params. 
        # We need to actually update curr_state outside of the gradient check if we want 'online' feel
        # But update_step essentially ran the steps to get the loss.
        # We can just re-run or rely on the fact that for tuning we valid on short bursts.
        # Let's just re-run 100 steps with new R to advance the "world".
        
        # Actually simplest is just to reset state every time to see what R producing 17Sv from rest?
        # No, AMOC takes time to build up.
        # We should tune on a "spun up" state?
        # Let's assume we want to control the trajectory. 
        # Advancing state:
        
        curr_state, _ = jax.lax.scan(
            lambda c, _: (main.step_coupled(c, params, regridder, r_drag=current_r), None),
            curr_state, None, length=steps_per_epoch // 10 # Just advance a bit
        )
        
        print(f"{epoch:<5} | {loss:.4f}     | {amoc:.4f}     | {current_r:.5f}")

    print("\nTuning Complete.")
    final_r = jax.nn.softplus(raw_r_param)
    print(f"Optimal r_drag: {final_r:.6f}")

if __name__ == "__main__":
    run_tuning()
