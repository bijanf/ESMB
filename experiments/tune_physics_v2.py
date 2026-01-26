

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
W_TEMP = 1.0 
W_KE = 0.01 
W_ROUGHNESS = 100.0 # High penalty for channel noise (waves)


def loss_fn(params_tuple, state_init, params, regridder, steps=50):
    """
    Compute loss for given physics parameters.
    params_tuple: (raw_r_param, raw_kappa_param)
    """
    raw_r, raw_kgm, raw_kh, raw_kbi = params_tuple
    
    # Softplus to ensure > 0
    r_drag = jax.nn.softplus(raw_r) + 1.0e-4
    kappa_gm = jax.nn.softplus(raw_kgm) + 1.0
    kappa_h = jax.nn.softplus(raw_kh) + 10.0
    kappa_bi = jax.nn.softplus(raw_kbi) * 1.0e13 + 1.0e12 # Bi-harmonic scale
    
    # Checkpoint the step function
    @jax.checkpoint
    def step_fn_checkpointed(carry, _):
        s = carry
        new_s = main.step_coupled(s, params, regridder, r_drag=r_drag, kappa_gm=kappa_gm, kappa_h=kappa_h, kappa_bi=kappa_bi)
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

    # 4. Roughness (Noise) Penalty
    # Sum of squared gradients of SST x, y
    sst = final_state.fluxes.sst
    d_sst_x = sst[:, 1:] - sst[:, :-1]
    d_sst_y = sst[1:, :] - sst[:-1, :]
    roughness = jnp.mean(d_sst_x**2) + jnp.mean(d_sst_y**2)

    loss_amoc = (amoc_val - TARGET_AMOC)**2
    loss_temp = (t_mean - TARGET_TEMP)**2
    loss_ke = ke * 1000.0
    loss_rough = roughness * 1000.0 
    
    total_loss = W_AMOC * loss_amoc + W_TEMP * loss_temp + W_KE * loss_ke + W_ROUGHNESS * loss_rough

    return total_loss, (amoc_val, t_mean, ke, roughness, r_drag, kappa_gm, kappa_h, kappa_bi, final_state)



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
    # 2. kappa_gm (Start 1000)
    raw_kgm = 1000.0
    # 3. kappa_h (Start 500)
    raw_kh = 500.0
    # 4. kappa_bi (Start 1e13 scaled) -> raw 1.0
    raw_kbi = 1.0 
    
    params_tuple = (jnp.array(raw_r), jnp.array(raw_kgm), jnp.array(raw_kh), jnp.array(raw_kbi))
    
    learning_rate = 0.0001
    
    steps_per_update = 24 # 6 Hours
    updates_per_epoch = 120 # 30 Days
    n_epochs = 360 # 30 Years
    
    print(f"Tuning Config: Online Training (TBPTT)")
    print(f"LR={learning_rate}")
    print(f"Update Horizon: {steps_per_update} steps")
    print(f"Update Horizon: {steps_per_update} steps")
    print(f"{'Ep':<4} | {'Loss':<8} | {'AMOC':<6} | {'Temp':<6} | {'Rough':<6} | {'r_drag':<8} | {'k_gm':<6} | {'k_h':<6} | {'k_bi':<6}")
    print("-" * 100)
    print("-" * 85)
    
    @jax.jit
    def update_step(curr_params, state_start):
        (loss, (amoc, tm, ke, rough, current_r, current_kgm, current_kh, current_kbi, final_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            curr_params, state_start, params, regridder, steps=steps_per_update
        )
        
        # Unpack Grads
        grad_r, grad_kgm, grad_kh, grad_kbi = grads
        
        # Gradient Clipping
        grad_r = jnp.clip(grad_r, -0.1, 0.1)
        grad_kgm = jnp.clip(grad_kgm, -1.0, 1.0)
        grad_kh = jnp.clip(grad_kh, -1.0, 1.0)
        grad_kbi = jnp.clip(grad_kbi, -0.1, 0.1) # Scaled
        
        # Check NaNs
        is_finite = jnp.isfinite(grad_r) & jnp.isfinite(grad_kgm) & jnp.isfinite(loss)
        
        safe_grad_r = jnp.where(is_finite, grad_r, 0.0)
        safe_grad_kgm = jnp.where(is_finite, grad_kgm, 0.0)
        safe_grad_kh = jnp.where(is_finite, grad_kh, 0.0)
        safe_grad_kbi = jnp.where(is_finite, grad_kbi, 0.0)
        
        # Updates
        new_raw_r = curr_params[0] - learning_rate * safe_grad_r
        new_raw_kgm = curr_params[1] - learning_rate * safe_grad_kgm
        new_raw_kh = curr_params[2] - learning_rate * safe_grad_kh
        new_raw_kbi = curr_params[3] - learning_rate * safe_grad_kbi
        
        return (new_raw_r, new_raw_kgm, new_raw_kh, new_raw_kbi), loss, amoc, tm, current_r, current_kgm, current_kh, current_kbi, final_state
 
    curr_state = state
    curr_r_k = params_tuple
    
    for epoch in range(n_epochs):
        t0 = time.time()
        
        epoch_loss = 0.0
        
        for _ in range(updates_per_epoch):
            new_params, loss, amoc, tm, r_val, kgm_val, kh_val, kbi_val, next_state = update_step(curr_r_k, curr_state)
            curr_r_k = new_params
            
            # Unpack Roughness from aux? Wait, update_step returns unpacked aux now
            # Actually update_step signature changed.
            # (new_params), loss, amoc, tm, current_r, current_kgm, current_kh, current_kbi, next_state
            
            curr_state = next_state
            
            epoch_loss += float(loss)
            
        epoch_loss /= updates_per_epoch
        t1 = time.time()
        
        print(f"{epoch:<4} | {epoch_loss:<8.2f} | {float(amoc):<6.2f} | {float(tm):<6.2f} | {float(r_val):<8.5f} | {float(kgm_val):<6.1f} | {float(kh_val):<6.1f} | {float(kbi_val):.1e}")
        
    print("Tuning Complete.")
    final_r = jax.nn.softplus(curr_r_k[0]) + 1.0e-4
    final_kgm = jax.nn.softplus(curr_r_k[1]) + 1.0
    final_kh = jax.nn.softplus(curr_r_k[2]) + 10.0
    final_kbi = jax.nn.softplus(curr_r_k[3]) * 1.0e13 + 1.0e12

    print(f"Optimal r_drag: {final_r:.6f}")
    print(f"Optimal kappa_gm: {final_kgm:.2f}")
    print(f"Optimal kappa_h: {final_kh:.2f}")
    print(f"Optimal kappa_bi: {final_kbi:.2e}")


if __name__ == "__main__":
    run_tuning()
