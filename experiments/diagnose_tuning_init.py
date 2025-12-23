
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import main, data
from chronos_esm.ocean import diagnostics

# Enable NaN debugging
jax.config.update("jax_debug_nans", False) 

def check_array(name, arr):
    arr_np = np.array(arr)
    if np.any(np.isnan(arr_np)):
        print(f"FAILED: {name} contains NaNs!")
        return False
    if np.any(np.isinf(arr_np)):
        print(f"FAILED: {name} contains Infs!")
        return False
    print(f"OK: {name:<20} | Min: {arr_np.min():.4e} | Max: {arr_np.max():.4e} | Mean: {arr_np.mean():.4e}")
    return True

def run_diagnostic():
    print("Running Diagnostic on Tuned Control Init...")
    
    # Setup (same as tune_physics_v2.py)
    mask = jnp.array(data.load_bathymetry_mask())
    params = main.ModelParams(mask=mask, co2_ppm=280.0)
    state = main.init_model()
    regridder = main.regrid.Regridder()
    
    # 1. Spinup
    print("\n--- Phase 1: Spinup (Static) ---")
    defaults_r = 1.0e-2
    spinups = 350
    
    @jax.jit
    def spinup(s):
        s, _ = jax.lax.scan(lambda c, _: (main.step_coupled(c, params, regridder, r_drag=defaults_r), None), s, None, length=spinups)
        return s
        
    print(f"Running {spinups} steps of spinup...")
    state = spinup(state)
    jit_status = state.ocean.temp.block_until_ready()
    print("Spinup Complete.")
    
    # 2. Inspect State
    print("\n--- Phase 2: State Inspection ---")
    valid = True
    valid &= check_array("Ocean U", state.ocean.u)
    valid &= check_array("Ocean V", state.ocean.v)
    valid &= check_array("Ocean Temp", state.ocean.temp)
    valid &= check_array("Ocean Salt", state.ocean.salt)
    valid &= check_array("Ocean Psi", state.ocean.psi)
    valid &= check_array("Atmos Temp", state.atmos.temp)
    valid &= check_array("Atmos U", state.atmos.u)
    
    if not valid:
        print("CRITICAL: State corrupted during spinup. Aborting.")
        return

    # 3. Test Gradients
    print("\n--- Phase 3: Gradient Test ---")
    
    # Define Loss (Simplified copy from tune_physics_v2)
    TARGET_AMOC = 17.0
    TARGET_TEMP = 288.0
    
    def run_grad_test(steps_test):
        print(f"\nTesting Horizon: {steps_test} steps...")
        
        def loss_fn(raw_r_param, state_init):
            r_drag = jax.nn.softplus(raw_r_param) + 1.0e-4
            
            # Checkpoint the step function to allow backprop
            @jax.checkpoint
            def step_fn_checkpointed(carry, _):
                s = carry
                new_s = main.step_coupled(s, params, regridder, r_drag=r_drag)
                return new_s, None

            final_state, _ = jax.lax.scan(step_fn_checkpointed, state_init, None, length=steps_test)
            
            moc = diagnostics.compute_moc(final_state.ocean)
            nlat = moc.shape[1]
            lat_idx = min(int((30.0 - (-90.0)) / 180.0 * nlat), nlat - 1)
            amoc_val = jnp.max(moc[:, lat_idx])
            t_mean = jnp.mean(final_state.atmos.temp)
            
            loss = (amoc_val - TARGET_AMOC)**2 + (t_mean - TARGET_TEMP)**2
            return loss, (amoc_val, t_mean)

        initial_r = 1.0e-2
        raw_r_param = jnp.array(np.log(np.exp(initial_r) - 1.0))
        
        try:
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(raw_r_param, state)
            print(f"Loss: {loss}")
            print(f"AMOC: {aux[0]}")
            print(f"Grads: {grads}")
            
            if jnp.isnan(grads) or jnp.isnan(loss):
                print(f"FAILURE: Gradients/Loss are NaN at {steps_test} steps!")
                return False
            else:
                print(f"SUCCESS: Gradients are finite at {steps_test} steps.")
                return True
                
        except Exception as e:
            print(f"CRASH at {steps_test} steps: {e}")
            return False

    # Test 10 steps first
    # if run_grad_test(10):
    #     # Test 24 steps (Proposed Horizon)
    #     run_grad_test(24)
    #     # Test 96 steps (Target)
    #     run_grad_test(96)

    print("\n--- Phase 4: Sequential Update Test (Simulation of Tuning Loop) ---")
    
    # Run a sequence of updates to see when it breaks
    lr = 0.0001
    current_state = state
    raw_r_param = jnp.array(np.log(np.exp(1.0e-2) - 1.0))
    
    # Define update step (same as tuning script)
    @jax.jit
    def update_step(raw_r, state_start):
        def loss_fn(r_p, s_init):
             r_d = jax.nn.softplus(r_p) + 1.0e-4
             # Checkpoint for memory efficiency
             @jax.checkpoint
             def step_fn_chk(carry, _):
                 return main.step_coupled(carry, params, regridder, r_drag=r_d), None
             
             final_s, _ = jax.lax.scan(step_fn_chk, s_init, None, length=24) # 24 steps = 6 hours
             
             # Metrics
             moc = diagnostics.compute_moc(final_s.ocean)
             nlat = moc.shape[1]
             lat_idx = min(int((30.0 - (-90.0)) / 180.0 * nlat), nlat - 1)
             amoc_val = jnp.max(moc[:, lat_idx])
             t_mean = jnp.mean(final_s.atmos.temp)
             ke = jnp.mean(final_s.ocean.u**2 + final_s.ocean.v**2)
             
             loss = (amoc_val - TARGET_AMOC)**2 + (t_mean - TARGET_TEMP)**2
             return loss, (amoc_val, t_mean, ke, final_s)

        (loss, (amoc, tm, ke, next_s)), grads = jax.value_and_grad(loss_fn, has_aux=True)(raw_r, state_start)
        
        # Clip grads
        grads = jnp.clip(grads, -0.5, 0.5)
        new_raw_r = raw_r - lr * grads
        
        return new_raw_r, loss, amoc, tm, ke, grads, next_s

    print(f"{'Step':<5} | {'Loss':<10} | {'AMOC':<10} | {'r_drag':<10} | {'Grad':<10}")
    
    for i in range(50): # Run 50 updates
        try:
             # Force Wait to catch NaN immediately
             raw_r_param, loss, amoc, tm, ke, grads, current_state = update_step(raw_r_param, current_state)
             
             # Block until ready to detect crashes
             loss.block_until_ready()
             
             current_r = float(jax.nn.softplus(raw_r_param) + 1.0e-4)
             print(f"{i:<5} | {float(loss):<10.2f} | {float(amoc):<10.2f} | {current_r:<10.5f} | {float(grads):.2e}")
             
             if np.isnan(float(loss)) or np.isnan(float(grads)):
                 print(f"FAILED: NaN detected at step {i}")
                 break
                 
        except Exception as e:
            print(f"CRASH at step {i}: {e}")
            break

if __name__ == "__main__":
    run_diagnostic()
