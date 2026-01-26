
import os
import sys
import jax
import jax.numpy as jnp
import numpy as np
from chronos_esm import main, data
from chronos_esm.ocean import diagnostics

def reproduce():
    print("Loading mask...")
    mask = jnp.array(data.load_bathymetry_mask())
    print(f"Mask shape: {mask.shape}")
    print(f"Ocean points: {jnp.sum(mask)}")
    print(f"Fraction: {jnp.mean(mask)}")

    print("Initializing model...")
    params = main.ModelParams(mask=mask)
    state = main.init_model()
    regridder = main.regrid.Regridder()

    # Check initial V (Should be 0)
    print(f"Initial Max V: {jnp.max(jnp.abs(state.ocean.v))}")

    # Step coupled
    print("Stepping model (1 step)...")
    # Using default r_drag (0.05) or whatever step_coupled uses
    # Note: main.py HARDCODES r_drag=0.002 in step_ocean call currently
    # We want to see if this hardcoded value produces 0 AMOC.
    
    new_state = main.step_coupled(state, params, regridder)
    
    # Check V after 1 step
    v = new_state.ocean.v
    print(f"Step 1 Max V: {jnp.max(jnp.abs(v))}")
    print(f"Step 1 Mean V: {jnp.mean(v)}")
    
    # Compute AMOC
    amoc = diagnostics.compute_amoc_index(new_state.ocean)
    print(f"Step 1 AMOC: {amoc} Sv")
    
    # Run 50 steps
    print("Running 50 steps loop...")
    
    def scan_fn(carry, _):
        s = carry
        s = main.step_coupled(s, params, regridder)
        return s, None
        
    final_state, _ = jax.lax.scan(scan_fn, state, None, length=50)
    
    v_final = final_state.ocean.v
    print(f"Step 50 Max V: {jnp.max(jnp.abs(v_final))}")
    amoc_final = diagnostics.compute_amoc_index(final_state.ocean)
    print(f"Step 50 AMOC: {amoc_final} Sv")

if __name__ == "__main__":
    reproduce()
