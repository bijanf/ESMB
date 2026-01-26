
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
RESUME_FILE = OUTPUT_DIR / "year_035.nc"

def run_resume_safe():
    print("=== CENTURY RUN RESUME (ATTEMPT 4): SAFE TIMESTEP (30s) ===")
    
    script_content = f"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from chronos_esm import main, io as model_io, data, log_utils, config
import jax.numpy as jnp

# Load Year 35 State (Verified Healthy)
if not Path("{RESUME_FILE}").exists():
    print("Error: Resume file not found!")
    sys.exit(1)
    
print("Loading resume state from {RESUME_FILE}...")
state = model_io.load_state_from_netcdf("{RESUME_FILE}")

# Calculate Steps for remaining 65 Years
years_to_run = 65.0
dt_coupled = 900.0 
dt_atmos = 30.0   # SAFE BASELINE (Was 150s, 120s, 90s -> All Failed)

total_steps = int(years_to_run * 365 * 24 * 3600 / dt_coupled)
steps_per_year = int(365 * 24 * 3600 / dt_coupled)

params = main.ModelParams(co2_ppm=280.0, solar_constant=1361.0, mask=data.load_bathymetry_mask(nz=15))
reg = main.regrid.Regridder()
PHYS_PARAMS = {{"qc_ref": 0.4915, "epsilon_smooth": 4.71e-2}}

output_dir = Path("{OUTPUT_DIR}")
logger = log_utils.setup_logger("century_resume_safe", output_dir / "model_resume_safe.log")

start_time = time.time()

print(f"Running w/ DT_ATMOS={{dt_atmos}}... (This will be slower but stable)")

for step in range(1, total_steps + 1):
    # Standard Physics (No need for extreme viscosity if DT is small enough)
    # We verify if DT=30 is enough.
    state = main.step_coupled(
        state, params, reg, 
        physics_params=PHYS_PARAMS,
        Ah=5.0e5,           # Standard Viscosity
        shapiro_strength=0.1, # Standard Filter
        kappa_gm=1000.0
    )
    
    # Stability Check
    if step % 500 == 0:
        elapsed = time.time() - start_time
        rate = step / elapsed
        state.ocean.temp.block_until_ready()
        print(f"Step {{step}}/{{total_steps}} ({{rate:.2f}} steps/s)")
        
        # Check Atmos Max Wind to catch explosions early
        u_max = float(jnp.max(jnp.abs(state.atmos.u)))
        if u_max > 100.0 or jnp.isnan(u_max):
             logger.error(f"Instability Detected! Max Wind={{u_max:.1f}} m/s")
             sys.exit(1)

    # Save Yearly
    if step % steps_per_year == 0:
        year_idx = 35 + (step // steps_per_year) 
        fname = output_dir / f"year_{{year_idx:03d}}.nc"
        model_io.save_state_to_netcdf(state, fname)
        t_mean = float(state.atmos.temp.mean())
        logger.info(f"Year {{year_idx}}: Global T={{t_mean:.2f}}")

print("Century Run Resume Complete.")
model_io.save_state_to_netcdf(state, output_dir / "final_state_100y.nc")
"""

    runner_script = OUTPUT_DIR / "run_core_resume_safe.py"
    with open(runner_script, "w") as f:
        f.write(script_content)
        
    env = os.environ.copy()
    env["CHRONOS_DT_ATMOS"] = "30.0" 
    
    subprocess.run([sys.executable, str(runner_script)], env=env, check=True)

if __name__ == "__main__":
    run_resume_safe()
