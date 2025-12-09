
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from chronos_esm import main
from chronos_esm.config import DT_ATMOS
from chronos_esm.ocean import diagnostics as ocean_diagnostics

def run_verification(years: float = 1.0):
    """
    Run the model for a specified number of years and track stability metrics.
    """
    print(f"Starting Stability Verification Run for {years} years...")
    
    # Time settings
    seconds_per_year = 365 * 24 * 3600
    total_seconds = years * seconds_per_year
    steps = int(total_seconds / DT_ATMOS)
    
    print(f"Total Steps: {steps}")
    print(f"Time Step (Atmos): {DT_ATMOS} s")
    
    # Initialize
    state = main.init_model()
    regridder = main.regrid.Regridder()
    params = main.ModelParams()
    
    # Use the already jitted step function
    step_fn = main.step_coupled
    
    # Metrics storage
    history = {
        'time': [],
        'gmst': [],
        'max_temp': [],
        'min_temp': [],
        'max_wind': [],
        'amoc': []
    }
    
    # Run Loop
    # We'll run in chunks to log progress
    chunk_size = 100 # Steps per chunk
    num_chunks = steps // chunk_size
    
    start_time = time.time()
    
    for i in range(num_chunks):
        # Run a chunk
        # We can use lax.scan for the chunk for speed
        def scan_fn(carry, _):
            s = carry
            new_s = step_fn(s, params, regridder)
            return new_s, None
            
        state, _ = jax.lax.scan(scan_fn, state, jnp.arange(chunk_size))
        
        # Compute Diagnostics (on host)
        # Block until ready to ensure we catch NaNs early
        state.atmos.temp.block_until_ready()
        
        # 1. Global Mean Surface Temp (Atmos)
        # Simple mean for now (should be area-weighted)
        gmst = float(jnp.mean(state.atmos.temp))
        
        # 2. Extremes
        max_t = float(jnp.max(state.atmos.temp))
        min_t = float(jnp.min(state.atmos.temp))
        
        # 3. Winds
        wind_speed = jnp.sqrt(state.atmos.u**2 + state.atmos.v**2)
        max_wind = float(jnp.max(wind_speed))
        
        # 4. AMOC
        amoc = float(ocean_diagnostics.compute_amoc_index(state.ocean))
        
        # Log
        current_time_days = (i + 1) * chunk_size * DT_ATMOS / (24 * 3600)
        
        history['time'].append(current_time_days)
        history['gmst'].append(gmst)
        history['max_temp'].append(max_t)
        history['min_temp'].append(min_t)
        history['max_wind'].append(max_wind)
        history['amoc'].append(amoc)
        
        if (i + 1) % 10 == 0:
            print(f"Day {current_time_days:.1f}: GMST={gmst:.2f}K, Range=[{min_t:.1f}, {max_t:.1f}]K, MaxWind={max_wind:.1f}m/s, AMOC={amoc:.2f}Sv")
            
        # Check for instability
        if jnp.isnan(gmst) or max_wind > 200.0 or max_t > 360.0:
            print("!!! INSTABILITY DETECTED !!!")
            print(f"Stopping at Day {current_time_days:.1f}")
            break
            
    end_time = time.time()
    print(f"Run finished in {end_time - start_time:.2f} seconds")
    
    # Plotting
    try:
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Temp
        axes[0].plot(history['time'], history['gmst'], label='GMST')
        axes[0].fill_between(history['time'], history['min_temp'], history['max_temp'], alpha=0.2, label='Range')
        axes[0].set_ylabel('Temperature [K]')
        axes[0].set_title('Global Temperature Stability')
        axes[0].legend()
        
        # Wind
        axes[1].plot(history['time'], history['max_wind'], color='orange')
        axes[1].set_ylabel('Max Wind Speed [m/s]')
        axes[1].set_title('Atmospheric Stability (Wind)')
        
        # AMOC
        axes[2].plot(history['time'], history['amoc'], color='green')
        axes[2].set_ylabel('AMOC [Sv]')
        axes[2].set_title('Ocean Circulation (AMOC)')
        axes[2].set_xlabel('Time [Days]')
        
        plt.tight_layout()
        plt.savefig('stability_report.png')
        print("Saved stability_report.png")
    except Exception as e:
        print(f"Could not save plot: {e}")


if __name__ == "__main__":
    # Run for 5 days
    run_verification(years=5.0/365.0)
