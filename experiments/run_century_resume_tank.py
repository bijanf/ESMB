
import sys
import time
from pathlib import Path
import os
import subprocess

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import main, io as model_io, data, log_utils, config
import jax.numpy as jnp

OUTPUT_DIR = Path("outputs/century_run")
RESUME_FILE = OUTPUT_DIR / "year_042.nc"  # FIXED: Use verified safe state, not poisoned year_046

def run_resume_tank():
    print("=== CENTURY RUN RESUME (ATTEMPT 10): STABILIZED TANK MODE (Year 42, Ah=5e6, DT=30s) ===")
    
    script_content = """
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from chronos_esm import main, io as model_io, data, log_utils, config
import jax.numpy as jnp

# Load Year 42 State (VERIFIED SAFE - Year 46/47 are poisoned with hidden instability)
if not Path("outputs/century_run/year_042.nc").exists():
    print("Error: Resume file not found!")
    sys.exit(1)

print("Loading resume state from outputs/century_run/year_042.nc...")
state = model_io.load_state_from_netcdf("outputs/century_run/year_042.nc")

# Calculate Steps for remaining 58 Years (100 - 42 = 58)
years_to_run = 58.0
dt_coupled = 900.0 
dt_atmos = 30.0   # TANK MODE TIMESTEP (30s)

total_steps = int(years_to_run * 365 * 24 * 3600 / dt_coupled)
steps_per_year = int(365 * 24 * 3600 / dt_coupled)

params = main.ModelParams(co2_ppm=280.0, solar_constant=1361.0, mask=data.load_bathymetry_mask(nz=15))
reg = main.regrid.Regridder()
PHYS_PARAMS = {"qc_ref": 0.4915, "epsilon_smooth": 4.71e-2}

output_dir = Path("outputs/century_run")
logger = log_utils.setup_logger("century_resume_tank", output_dir / "model_resume_tank.log")

start_time = time.time()

print(f"Running w/ Ah=5.0e6, DT=30s (Tank Mode)...")

for step in range(1, total_steps + 1):
    # TANK MODE OPTION:
    # Ah=5.0e6 (Max Visc) + DT=30s (Min Timestep)
    state = main.step_coupled(
        state, params, reg, 
        physics_params=PHYS_PARAMS,
        Ah=5.0e6,           # MAX VISCOSITY
        shapiro_strength=0.9, # EXTREME FILTER
        kappa_gm=3000.0,
        kappa_h=2000.0
    )
    
    # Stability Check - Enhanced monitoring for early warning
    if step % 500 == 0:
        elapsed = time.time() - start_time
        rate = step / elapsed
        state.ocean.temp.block_until_ready()

        t_global = float(state.atmos.temp.mean())
        max_wind = float(jnp.sqrt(state.atmos.u**2 + state.atmos.v**2).max())
        max_vort = float(jnp.abs(state.atmos.vorticity).max())

        print(f"Step {step}/{total_steps} ({rate:.2f} steps/s) | T={t_global:.1f}K | MaxWind={max_wind:.1f}m/s | MaxVort={max_vort:.2e}")

        # NaN check
        if jnp.isnan(t_global):
             logger.error("NaN Detected in Tank Resume!")
             sys.exit(1)

        # Early warning: Wind approaching danger zone
        if max_wind > 80.0:
             logger.warning(f"WARNING: High wind detected ({max_wind:.1f} m/s) - approaching instability!")
        if max_wind > 120.0:
             logger.error(f"CRITICAL: Supersonic winds ({max_wind:.1f} m/s) - saving emergency checkpoint!")
             model_io.save_state_to_netcdf(state, output_dir / "emergency_checkpoint.nc")
             sys.exit(1)

    # Save Yearly
    if step % steps_per_year == 0:
        year_idx = 42 + (step // steps_per_year)  # Start from Year 42
        fname = output_dir / f"year_{year_idx:03d}.nc"
        model_io.save_state_to_netcdf(state, fname)
        t_mean = float(state.atmos.temp.mean())
        logger.info(f"Year {year_idx}: Global T={t_mean:.2f}")

print("Century Run Tank Resume Complete.")
model_io.save_state_to_netcdf(state, output_dir / "final_state_100y.nc")
"""

    runner_script = OUTPUT_DIR / "run_core_resume_tank.py"
    with open(runner_script, "w") as f:
        f.write(script_content)
        
    env = os.environ.copy()
    env["CHRONOS_DT_ATMOS"] = "30.0" 
    
    subprocess.run([sys.executable, str(runner_script)], env=env, check=True)

if __name__ == "__main__":
    run_resume_tank()
