
import sys
import os
import shutil
from pathlib import Path
import jax.numpy as jnp
import jax
import time
import numpy as np
import optax
import matplotlib.pyplot as plt

# Add parent path
sys.path.insert(0, os.path.abspath(os.curdir))

from chronos_esm import main, data
from chronos_esm import io as model_io
from chronos_esm.config import ATMOS_GRID, DT_ATMOS
from chronos_esm.atmos import physics
from chronos_esm.coupler import regrid

OUTPUT_DIR = Path("outputs/tuning_results")
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir()

def get_target_precip(ny=ATMOS_GRID.nlat):
    """
    Define a simple idealized zonal mean precipitation target [mm/day].
    ITCZ peak at Equator (~8 mm/day), Minima in Subtropics (~1 mm/day), Increase in Midlats.
    """
    lat = np.linspace(-90, 90, ny)
    # Gaussian ITCZ
    itcz = 8.0 * np.exp(-(lat / 10.0)**2)
    # Midlat storm tracks
    midlat = 3.0 * np.exp(-( (np.abs(lat) - 45.0) / 15.0 )**2)
    
    # Combined
    target_zonal = itcz + midlat
    return jnp.array(target_zonal)

def compute_loss(params_dict, state, regridder, target_zonal):
    """
    Run model forward for N steps and compute loss against target.
    """
    # Run 1 Ocean Step (30 Atmos Steps)
    # Use differentiable `main.step_coupled`
    
    # We assume 'state' is the starting state (constant in this grad step)
    
    # We need to run enough steps to establish a response, but keep it short for gradients.
    # 1 Ocean Step (900s) should use the new parameters immediately in atmos.
    
    # Reuse default dynamics params
    new_state = main.step_coupled(
        state, 
        main.ModelParams(co2_ppm=280.0), 
        regridder,
        physics_params=params_dict
    )
    
    # Extract Global Mean Precip field (Acc/Avg)
    precip_field = new_state.fluxes.precip # [kg/m2/s]
    
    # Convert to mm/day
    precip_mm_day = precip_field * 86400.0
    
    # Zonal Mean
    precip_zonal = jnp.mean(precip_mm_day, axis=1)
    
    # Loss: MSE against target
    loss = jnp.mean((precip_zonal - target_zonal)**2)
    
    # Regularization (Penalize extreme QC_REF)
    qc_ref = params_dict['qc_ref']
    reg = 100.0 * (jnp.maximum(0.0, 0.1 - qc_ref) + jnp.maximum(0.0, qc_ref - 0.9)) # Barrier at 0.1 and 0.9
    
    # Also penalize if Global Mean T drifts too far? Maybe not for 1 step.
    
    return loss + reg

def run_tuning():
    print("Finding latest restart file...")
    source_dir = Path("outputs/production_control")
    restarts = sorted(list(source_dir.glob("restart_*.nc")))
    latest_restart = restarts[-1]
    print(f"Loading {latest_restart}")
    
    # Load state (with rescue logic implied by IO fix)
    state = model_io.load_state_from_netcdf(latest_restart)
    regridder = main.regrid.Regridder()
    
    target_zonal = get_target_precip()
    
    # Initial Params
    # qc_ref: 0.30
    # epsilon: 1e-2
    
    params = {
        "qc_ref": 0.30,
        "epsilon_smooth": 1.0e-2
    }
    
    # Optimizer
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)
    
    # Value and Grad Function
    val_and_grad_fn = jax.value_and_grad(compute_loss, argnums=0)
    
    print(f"Starting Tuning...")
    print(f"Target Zonal Mean Max: {jnp.max(target_zonal):.2f} mm/day")
    
    history = []
    
    for i in range(20):
        loss_val, grads = val_and_grad_fn(params, state, regridder, target_zonal)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        # Simple bounds clipping manually if needed, or rely on regularization
        params['qc_ref'] = jnp.clip(params['qc_ref'], 0.1, 0.9)
        params['epsilon_smooth'] = jnp.clip(params['epsilon_smooth'], 1e-4, 1.0)
        
        print(f"Step {i}: Loss={loss_val:.4f}, QC_REF={params['qc_ref']:.4f}, Eps={params['epsilon_smooth']:.4e}")
        history.append((loss_val, params['qc_ref'], params['epsilon_smooth']))
        
    # Plot history
    hist = np.array(history)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(hist[:, 0])
    ax[0].set_title("Loss")
    ax[1].plot(hist[:, 1])
    ax[1].set_title("QC_REF")
    ax[2].plot(hist[:, 2])
    ax[2].set_title("Epsilon")
    plt.savefig(OUTPUT_DIR / "tuning_history.png")
    
    print("-" * 30)
    print("TUNING COMPLETE")
    print(f"Best Params: QC_REF={params['qc_ref']:.4f}, Eps={params['epsilon_smooth']:.4e}")
    print("-" * 30)

if __name__ == "__main__":
    run_tuning()
