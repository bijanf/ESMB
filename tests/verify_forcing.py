
import jax
import jax.numpy as jnp
import numpy as np
from chronos_esm import main
import matplotlib.pyplot as plt

def verify_forcing():
    print("Initializing Model...")
    state = main.init_model()
    
    # Check Initial Winds (Should be small or zero depending on init)
    u_init = state.atmos.u
    print(f"Initial U Max: {jnp.max(u_init):.2f} m/s")
    
    print("Running for 5 days (480 steps)...")
    # DT_OCEAN = 900s. 5 days = 432000s. Steps = 480.
    
    steps = 480
    params = main.ModelParams()
    
    final_state = main.run_simulation(steps, params)
    
    # Check Final Winds
    u_final = final_state.atmos.u
    v_final = final_state.atmos.v
    t_final = final_state.atmos.temp
    
    u_max = jnp.max(u_final)
    u_min = jnp.min(u_final)
    v_max = jnp.max(v_final)
    t_max = jnp.max(t_final)
    t_min = jnp.min(t_final)
    
    print(f"Final U Range: {u_min:.2f} to {u_max:.2f} m/s")
    print(f"Final V Range: {jnp.min(v_final):.2f} to {jnp.max(v_final):.2f} m/s")
    print(f"Final T Range: {t_min:.2f} to {t_max:.2f} K")
    
    # Check for Gradient
    t_gradient = t_max - t_min
    print(f"Pole-to-Equator T Gradient check: {t_gradient:.2f} K")
    
    if u_max < 0.1 and u_min > -0.1:
        print("FAILURE: Winds are dead.")
    else:
        print("SUCCESS: Winds are active.")

    if t_gradient < 10.0:
        print("FAILURE: Temperature is flat.")
    else:
        print("SUCCESS: Temperature gradient exists.")
        
if __name__ == "__main__":
    verify_forcing()
