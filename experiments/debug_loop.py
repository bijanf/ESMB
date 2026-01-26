
import sys
from pathlib import Path
import jax.numpy as jnp
import jax

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import main, data
from chronos_esm.atmos import physics

def debug_loop():
    print("Initializing model...")
    state = main.init_model()
    regridder = main.regrid.Regridder()
    mask = data.load_bathymetry_mask()
    params = main.ModelParams(mask=mask, co2_increase_rate=0.0)
    
    # step_fn = jax.jit(main.step_coupled)
    step_fn = main.step_coupled
    
    current_state = state
    for i in range(10):
        print(f"\n--- Step {i+1} ---")
        
        # We need to manually handle the CO2 forcing part because check_loop.py calls step_coupled
        # But wait, step_coupled NOW has the CO2 forcing inside it!
        # So we just call it.
        
        try:
            new_state = step_fn(current_state, params, regridder)
            # Force evaluation
            t_mean = float(jnp.mean(new_state.atmos.temp))
            u_max = float(jnp.max(jnp.abs(new_state.atmos.u)))
            vort_max = float(jnp.max(jnp.abs(new_state.atmos.vorticity)))
            q_max = float(jnp.max(new_state.atmos.q))
            
            print(f"T_mean: {t_mean:.2f} K")
            print(f"U_max: {u_max:.2f} m/s")
            print(f"Vort_max: {vort_max:.2e}")
            print(f"Q_max: {q_max:.2e}")
            
            if jnp.isnan(t_mean) or jnp.isnan(u_max):
                 print("NaN detected!")
                 break
                 
            current_state = new_state
            
        except Exception as e:
            print(f"Crash: {e}")
            break

if __name__ == "__main__":
    debug_loop()
