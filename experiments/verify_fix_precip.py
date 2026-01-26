
import sys
import os
import shutil
from pathlib import Path
import jax.numpy as jnp
import jax
import time

# Add parent path
sys.path.insert(0, os.path.abspath(os.curdir))

from chronos_esm import main, data
from chronos_esm import io as model_io
from chronos_esm.config import DT_OCEAN

# Setup minimal env for verification run
OUTPUT_DIR = Path("outputs/verify_precip")
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir()

def run_verification():
    print("Finding latest restart file...")
    source_dir = Path("outputs/production_control")
    restarts = sorted(list(source_dir.glob("restart_*.nc")))
    if not restarts:
        print("No restart file found!")
        return
    
    latest_restart = restarts[-1]
    print(f"Resuming from {latest_restart}")
    
    # Load state
    state = model_io.load_state_from_netcdf(latest_restart)
    
    # Setup step function params (Reuse from production)
    mask = data.load_bathymetry_mask(nz=15)
    params = main.ModelParams(
        co2_ppm=280.0,
        solar_constant=1361.0,
        mask=mask
    )
    
    regridder = main.regrid.Regridder()

    # Run for a few steps to trigger physics
    # 1 ocean step (900s) processes multiple atmos steps (e.g. 30s)
    
    # DEBUG: Check Initial RH
    from chronos_esm.atmos import physics
    temp = state.atmos.temp
    ln_ps = state.atmos.ln_ps
    p = jnp.exp(ln_ps)
    q_sat = physics.compute_saturation_humidity(temp, p)
    rh = state.atmos.q / q_sat
    print(f"INITIAL Max RH: {float(jnp.max(rh)):.4f}")
    print(f"INITIAL Mean RH: {float(jnp.mean(rh)):.4f}")
    
    n_steps = 10 
    print(f"Running {n_steps} ocean steps...")
    
    current_state = state
    
    t0 = time.time()
    for i in range(n_steps):
        current_state = main.step_coupled(
            current_state, params, regridder,
            r_drag=5.0e-2,
            kappa_gm=1000.0,
            kappa_h=1000.0,
            Ah=2.0e5,
            shapiro_strength=0.1
        )
        current_state.fluxes.precip.block_until_ready()
        print(f"Step {i+1}/{n_steps} done.")
        
    t1 = time.time()
    print(f"Run finished in {t1-t0:.2f}s")
    
    # Diagnostic Check
    precip = current_state.fluxes.precip
    max_precip = float(jnp.max(precip))
    mean_precip = float(jnp.mean(precip))
    
    # Convert to mm/day
    max_mm_day = max_precip * 86400.0
    mean_mm_day = mean_precip * 86400.0
    
    print("-" * 30)
    print(f"VERIFICATION RESULTS")
    print("-" * 30)
    print(f"Max Precip: {max_mm_day:.4f} mm/day")
    print(f"Mean Precip: {mean_mm_day:.4f} mm/day")
    
    # DEBUG: Compute RH
    # Re-compute q_sat
    # Need pressure. Assume surface for now approx? 
    # Or use physics.compute_saturation_humidity(temp, p)
    # We can import physics
    from chronos_esm.atmos import physics
    
    temp = current_state.atmos.temp
    ln_ps = current_state.atmos.ln_ps
    p = jnp.exp(ln_ps)  # Corrected: ln_ps is ln(P_pascals)
    
    q_sat = physics.compute_saturation_humidity(temp, p)
    rh = current_state.atmos.q / q_sat
    
    print(f"Max RH: {float(jnp.max(rh)):.4f}")
    print(f"Mean RH: {float(jnp.mean(rh)):.4f}")
    print(f"Physics QC_REF: {physics.QC_REF}")
    
    print(f"Max Temp: {float(jnp.max(temp)):.2f} K")
    print(f"Mean Temp: {float(jnp.mean(temp)):.2f} K")
    print(f"Mean Pressure: {float(jnp.mean(p)):.2f} Pa")
    print(f"Mean ln_ps: {float(jnp.mean(ln_ps)):.2f}")
    print(f"Mean q: {float(jnp.mean(current_state.atmos.q)):.2e}")
    
    u = current_state.atmos.u
    v = current_state.atmos.v
    print(f"Max U: {float(jnp.max(jnp.abs(u))):.2f} m/s")
    print(f"Max V: {float(jnp.max(jnp.abs(v))):.2f} m/s")
    
    if max_mm_day > 0.1: # Threshold for 'raining'
        print("SUCCESS: Precipitation is triggering!")
        # Save output for plotting if needed
        model_io.save_state_to_netcdf(current_state, OUTPUT_DIR / "verify_state.nc")
    else:
        print("FAILURE: Precipitation still near zero.")

if __name__ == "__main__":
    run_verification()
