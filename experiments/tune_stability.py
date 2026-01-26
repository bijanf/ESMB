"""
Automated Tuning of Stability Parameters using JAX Gradients.

This script demonstrates how to find the optimal viscosity (Ah) and diffusion (kappa_bi)
parameters to minimize grid-scale noise ("roughness") while maintaining physical realism.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
# import optax  # Gradient processing (Removed)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import main, data
from chronos_esm import io as model_io

def compute_roughness(field):
    """Compute spatial roughness (variance of Laplacian)."""
    # Simple discrete Laplacian
    lap = (
        jnp.roll(field, 1, axis=0) + jnp.roll(field, -1, axis=0) +
        jnp.roll(field, 1, axis=1) + jnp.roll(field, -1, axis=1) -
        4 * field
    )
    return jnp.mean(lap**2)

def tune_parameters(steps=360, learning_rate=0.05, n_epochs=100):
    print("--- Starting Automated Parameter Tuning (Long Run with Checkpointing) ---")
    
    # 1. Load Initial State
    # Use a fresh state or load one
    try:
        mask = data.load_bathymetry_mask()
        params = main.ModelParams(mask=mask)
        # Initialize
        state_init = main.init_model()
        # Burn in a bit? No, start from cold or loaded.
        # Let's load the control run start if available
        ic_file = Path("outputs/production_control/state_0000.nc")
        if ic_file.exists():
            print(f"Loading IC from {ic_file}")
            state_init = model_io.load_state_from_netcdf(ic_file)
        else:
            print("Using fresh initial state.")
            
    except Exception as e:
        print(f"Setup failed: {e}")
        return

    # 2. Define Loss Function
    regridder = main.regrid.Regridder()
    
    # We optimize Log Parameters to ensure positivity
    # Initial guesses (Updated to near-stable manual values)
    # Ah ~ 2e5 -> log(2e5) ~ 12.2
    # kappa_bi ~ 1e16 -> log(1e16) ~ 36.8
    
    init_params = {
        "log_Ah": jnp.log(1.0e5), 
        "log_kappa_bi": jnp.log(5.0e14),
        "log_kappa_h": jnp.log(500.0)
    }
    
    # Manual Optimizer (Simple Gradient Descent with Momentum)
    # optimizer = optax.adam(learning_rate)
    # opt_state = optimizer.init(init_params)
    
    velocities = {k: 0.0 for k in init_params}
    beta = 0.9
    
    @jax.jit
    def loss_fn(trainable_params, start_state):
        # Decode parameters
        Ah = jnp.exp(trainable_params["log_Ah"])
        kappa_bi = jnp.exp(trainable_params["log_kappa_bi"])
        kappa_h = jnp.exp(trainable_params["log_kappa_h"])
        
        # Run Simulation
        @jax.checkpoint # Enable Rematerialization to save Memory
        def step(carry, _):
            s = carry
            # Fixed r_drag, kappa_gm for now
            new_s = main.step_coupled(
                s, params, regridder,
                r_drag=5.0e-2,
                kappa_gm=1000.0,
                kappa_h=kappa_h,
                kappa_bi=kappa_bi,
                Ah=Ah,
                Ab=0.0
            )
            return new_s, None
        
        final_state, _ = jax.lax.scan(step, start_state, jnp.arange(steps))
        
        # Calculate Metrics
        
        # 1. Roughness (SST Noise)
        # SST is in fluxes.sst
        sst = final_state.fluxes.sst
        roughness = compute_roughness(sst)
        
        # 2. Drift (Penalty for cooling/warming too fast)
        # Compare to start
        t_drift = jnp.mean((final_state.atmos.temp - start_state.atmos.temp)**2)
        
        # 3. Damping Penalty (Don't just turn viscosity to infinity)
        # Penalize extremely high values
        penalty = 1e-4 * (trainable_params["log_Ah"]**2 + trainable_params["log_kappa_bi"]**2)
        
        # Weighted Loss
        # We want Roughness < 0.1 ideally.
        # Drift should be small.
        total_loss = 1000.0 * roughness + 1.0 * t_drift # + penalty
        
        metrics = {
            "roughness": roughness,
            "drift": t_drift,
            "loss": total_loss,
            "Ah": Ah,
            "kappa_bi": kappa_bi
        }
        
        return total_loss, metrics

    # 3. Optimization Loop
    current_params = init_params
    
    print(f"{'Epoch':<6} | {'Loss':<10} | {'Roughness':<10} | {'Ah':<10} | {'Kappa_bi':<10}")
    print("-" * 60)
    
    for i in range(n_epochs):
        grads, metrics = jax.grad(loss_fn, has_aux=True)(current_params, state_init)
        
        # Manual Update (Momentum)
        for k in current_params:
            # Clip Gradients to prevent explosion
            g = jnp.clip(grads[k], -1.0, 1.0)
            velocities[k] = beta * velocities[k] + (1 - beta) * g
            current_params[k] = current_params[k] - learning_rate * velocities[k]
            
            # Hard Clamp Parameters to Safe Range
            if k == "log_kappa_bi":
                # Max 1e16 (approx 36.8)
                current_params[k] = jnp.clip(current_params[k], 0.0, 36.5)
            elif k == "log_Ah":
                 # Max 1e6 (approx 13.8)
                current_params[k] = jnp.clip(current_params[k], 0.0, 13.5)
        
        # updates, opt_state = optimizer.update(grads, opt_state)
        # current_params = optax.apply_updates(current_params, updates)
        
        print(f"{i:<6} | {metrics['loss']:<10.4f} | {metrics['roughness']:<10.6f} | {metrics['Ah']:<10.2e} | {metrics['kappa_bi']:<10.2e}")

    print("\n--- Optimized Parameters ---")
    print(f"Ah: {jnp.exp(current_params['log_Ah']):.4e}")
    print(f"Kappa_bi: {jnp.exp(current_params['log_kappa_bi']):.4e}")
    print(f"Kappa_h: {jnp.exp(current_params['log_kappa_h']):.4e}")
    
    return current_params

if __name__ == "__main__":
    tune_parameters()