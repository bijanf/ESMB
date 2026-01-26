
import sys
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import main, data
from chronos_esm.ocean import diagnostics

# Targets
TARGET_AMOC = 17.0

def compute_metrics(raw_r_param, state_init, params, regridder, steps=24):
    """
    Compute AMOC for a given r_drag parameter given a number of steps.
    """
    r_drag = jax.nn.softplus(raw_r_param) + 1.0e-4
    
    @jax.checkpoint
    def step_fn(carry, _):
        s = carry
        new_s = main.step_coupled(s, params, regridder, r_drag=r_drag)
        return new_s, None

    final_state, _ = jax.lax.scan(step_fn, state_init, None, length=steps)
    
    # Compute AMOC
    moc = diagnostics.compute_moc(final_state.ocean)
    nlat = moc.shape[1]
    lat_idx = int((30.0 - (-90.0)) / 180.0 * nlat)
    lat_idx = min(lat_idx, nlat - 1)
    amoc_val = jnp.max(moc[:, lat_idx])
    
    return amoc_val, final_state

def run_verification():
    print("Initializing Gradient Verification...")
    
    # Setup
    mask = jnp.array(data.load_bathymetry_mask())
    params = main.ModelParams(mask=mask, co2_ppm=280.0)
    state = main.init_model()
    regridder = main.regrid.Regridder()
    
    # Spinup slightly to get non-zero AMOC
    print("Spinning up for 100 steps...")
    defaults_r = 1.0e-2
    @jax.jit
    def spinup(s):
        s, _ = jax.lax.scan(lambda c, _: (main.step_coupled(c, params, regridder, r_drag=defaults_r), None), s, None, length=100)
        return s
    state = spinup(state)
    
    # Base Parameter
    initial_r = 1.0e-2
    raw_r = np.log(np.exp(initial_r) - 1.0)
    raw_r_param = jnp.array(raw_r)
    
    # 1. Compute AD Gradient (Short Horizon: 24 steps = 6 hours)
    steps_short = 24
    print(f"\n--- Checking Short Horizon ({steps_short} steps) ---")
    
    def loss_short(r):
        amoc, _ = compute_metrics(r, state, params, regridder, steps=steps_short)
        return (amoc - TARGET_AMOC)**2, amoc

    val, metrics = loss_short(raw_r_param)
    amoc_val = metrics
    grad_ad = jax.grad(lambda r: loss_short(r)[0])(raw_r_param)
    
    print(f"Base AMOC: {amoc_val:.4f} Sv")
    print(f"AD Gradient (dLoss/dr): {grad_ad:.6e}")
    
    # 2. Compute Finite Difference Gradient (Short Horizon)
    epsilon = 1e-4
    val_plus, _ = loss_short(raw_r_param + epsilon)
    val_minus, _ = loss_short(raw_r_param - epsilon)
    grad_fd = (val_plus - val_minus) / (2 * epsilon)
    
    print(f"FD Gradient (dLoss/dr): {grad_fd:.6e}")
    print(f"Relative Error: {abs((grad_ad - grad_fd)/grad_fd):.4f}")
    
    if np.sign(grad_ad) == np.sign(grad_fd):
        print("SUCCESS: AD and FD gradients match in sign for short horizon.")
    else:
        print("WARNING: AD and FD gradients have OPPOSITE signs for short horizon!")

    # 3. Check Long Term Sensitivity (FD only, as AD is too expensive)
    steps_long = 96 * 5 # 5 Days (approx 480 steps)
    print(f"\n--- Checking Longer Horizon ({steps_long} steps) ---")
    
    def get_amoc_long(r):
        # We just want AMOC, not full loss for now
        amoc, _ = compute_metrics(r, state, params, regridder, steps=steps_long)
        return amoc

    amoc_base = get_amoc_long(raw_r_param)
    amoc_plus = get_amoc_long(raw_r_param + epsilon)
    
    # Sensitivity d(AMOC)/d(r)
    # If r increases -> friction increases -> AMOC should DECREASE? Or increase?
    # Physically: Higher friction usually slows down circulation -> Lower AMOC.
    sensitivity = (amoc_plus - amoc_base) / epsilon
    
    print(f"Long-term Sensitivity d(AMOC)/d(raw_r): {sensitivity:.6e}")
    
    # Compare with Short-term Gradient implied sensitivity
    # Loss = (AMOC - Target)^2
    # dLoss/dr = 2 * (AMOC - Target) * dAMOC/dr
    # So dAMOC/dr = grad_ad / (2 * (AMOC - Target))
    
    implied_short_sensitivity = grad_ad / (2 * (amoc_val - TARGET_AMOC))
    print(f"Short-term Implied Sensitivity: {implied_short_sensitivity:.6e}")
    
    if np.sign(sensitivity) == np.sign(implied_short_sensitivity):
        print("SUCCESS: Short-term gradient direction matches Long-term sensitivity.")
    else:
        print("CRITICAL FAILURE: Short-term gradient points in OPPOSITE direction to Long-term effect.")
        print("Adjustment based on short-term gradient will likely destabilize the model.")

if __name__ == "__main__":
    run_verification()
