

import os
print("Python Process Started - Importing Modules...", flush=True)
import sys
import jax
import jax.numpy as jnp
import numpy as np
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import main, data
from chronos_esm.ocean import diagnostics


# Enable NaN debugging (False for stability)
# CRITICAL: Enabled for debugging job 5984475 crash
jax.config.update("jax_debug_nans", False)

# Targets
TARGET_AMOC = 17.0 # Sv
TARGET_TEMP = 288.0 # K (Global Mean Surface Temp)


# Weights
W_AMOC = 1.0
W_TEMP = 0.1 
W_KE = 0.01 # Kinetic Energy Penalty Weight


def loss_fn(params_tuple, state_init, params, regridder, steps=50):
    """
    Compute loss for given physics parameters.
    params_tuple: (raw_r_param, raw_kappa_param)
    """
    raw_r_param, raw_kappa_param = params_tuple
    
    # Softplus to ensure > 0
    r_drag = jax.nn.softplus(raw_r_param) + 1.0e-4
    kappa_gm = jax.nn.softplus(raw_kappa_param) + 1.0 # Min value 1.0
    
    # Checkpoint the step function
    @jax.checkpoint
    def step_fn_checkpointed(carry, _):
        s = carry
        new_s = main.step_coupled(s, params, regridder, r_drag=r_drag, kappa_gm=kappa_gm)
        return new_s, None

    final_state, _ = jax.lax.scan(step_fn_checkpointed, state_init, None, length=steps)
    
    # 1. Compute AMOC
    moc = diagnostics.compute_moc(final_state.ocean)
    nlat = moc.shape[1]
    lat_idx = min(int((30.0 - (-90.0)) / 180.0 * nlat), nlat - 1)
    amoc_val = jnp.max(moc[:, lat_idx])
    
    # 2. Global Mean Temp
    t_mean = jnp.mean(final_state.atmos.temp)

    # 3. Kinetic Energy
    ke = jnp.mean(final_state.ocean.u**2 + final_state.ocean.v**2)
    
    loss_amoc = (amoc_val - TARGET_AMOC)**2
    loss_temp = (t_mean - TARGET_TEMP)**2
    loss_ke = ke * 1000.0 # Scale up
    
    total_loss = W_AMOC * loss_amoc + W_TEMP * loss_temp + W_KE * loss_ke

    return total_loss, (amoc_val, t_mean, ke, r_drag, kappa_gm, final_state)



def run_tuning():
    print("Initializing JAX Auto-Tuning (v2) - Multi-Parameter...")
    print(f"Target AMOC: {TARGET_AMOC} Sv")
    print(f"Target Temp: {TARGET_TEMP} K")
    
    # Setup
    mask = jnp.array(data.load_bathymetry_mask())
    params = main.ModelParams(mask=mask, co2_ppm=280.0)
    state = main.init_model()
    regridder = main.regrid.Regridder()
    
    print("Spinning up for 1 year (static)...")
    defaults_r = 1.0e-2
    defaults_kappa = 1000.0
    spinups = 350
    
    @jax.jit
    def spinup(s):
        s, _ = jax.lax.scan(lambda c, _: (main.step_coupled(c, params, regridder, r_drag=defaults_r, kappa_gm=defaults_kappa), None), s, None, length=spinups)
        return s
        
    state = spinup(state)
    print("Spinup complete.")
    
    # Tuning Parameters
    # 1. r_drag
    raw_r = np.log(np.exp(1.0e-2) - 1.0)
    # 2. kappa_gm
    raw_kappa = np.log(np.exp(1000.0) - 1.0)
    
    params_tuple = (jnp.array(raw_r), jnp.array(raw_kappa))
    
    learning_rate = 0.0001
    
    steps_per_update = 24 # 6 Hours
    updates_per_epoch = 120 # 30 Days
    n_epochs = 360 # 30 Years
    
    print(f"Tuning Config: Online Training (TBPTT)")
    print(f"LR={learning_rate}")
    print(f"Update Horizon: {steps_per_update} steps")
    print(f"{'Month':<5} | {'Loss':<10} | {'AMOC':<10} | {'Temp':<10} | {'r_drag':<10} | {'kappa_gm':<10} (Grads)")
    print("-" * 85)
    
    @jax.jit
    def update_step(curr_params, state_start):
        (loss, (amoc, tm, ke, current_r, current_k, final_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            curr_params, state_start, params, regridder, steps=steps_per_update
        )
        
        # Unpack Grads
        grad_r, grad_k = grads
        
        # Gradient Clipping (Tightened for stability)
        # 0.1 clipping for stability
        grad_r = jnp.clip(grad_r, -0.1, 0.1)
        # Kappa gradients might be larger due to scale? Normalize?
        # Kappa is ~1000. dr/dloss might be small. 
        # Let's clip tightly too.
        grad_k = jnp.clip(grad_k, -1.0, 1.0) 
        
        # Check NaNs
        is_finite = jnp.isfinite(grad_r) & jnp.isfinite(grad_k) & jnp.isfinite(loss)
        
        safe_grad_r = jnp.where(is_finite, grad_r, 0.0)
        safe_grad_k = jnp.where(is_finite, grad_k, 0.0)
        
        # Updates
        new_raw_r = curr_params[0] - learning_rate * safe_grad_r
        # Kappa likely needs larger LR or separate LR? 
        # Using same LR first. Kappa ~ 1000. Log space -> raw ~ 7.
        # dL/dKappa might be small. 
        new_raw_k = curr_params[1] - learning_rate * safe_grad_k 
        
        return (new_raw_r, new_raw_k), loss, amoc, tm, current_r, current_k, (grad_r, grad_k), final_state
 
    curr_state = state
    curr_r_k = params_tuple
    
    for epoch in range(n_epochs):
        t0 = time.time()
        
        epoch_loss = 0.0
        
        for _ in range(updates_per_epoch):
            new_params, loss, amoc, tm, r_val, k_val, grads, next_state = update_step(curr_r_k, curr_state)
            curr_r_k = new_params
            curr_state = next_state
            
            epoch_loss += float(loss)
            
        epoch_loss /= updates_per_epoch
        t1 = time.time()
        
        # Print last step stats
        g_r, g_k = grads
        print(f"{epoch:<5} | {epoch_loss:<10.2f} | {float(amoc):<10.2f} | {float(tm):<10.2f} | {float(r_val):<10.5f} | {float(k_val):<10.1f} (Gr: {float(g_r):.1e}, {float(g_k):.1e}) [{t1-t0:.2f}s]")
        
    print("Tuning Complete.")
    final_r = jax.nn.softplus(curr_r_k[0]) + 1.0e-4
    final_k = jax.nn.softplus(curr_r_k[1]) + 1.0
    print(f"Optimal r_drag: {final_r:.6f}")
    print(f"Optimal kappa_gm: {final_k:.2f}")


if __name__ == "__main__":
    run_tuning()
