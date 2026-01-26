
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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Resume from 10-year run
RESUME_FILE = Path("outputs/long_run_fast/final_state.nc")

def run_century():
    print("=== CENTURY RUN: EXTENDING TO 100 YEARS ===")
    
    script_content = f"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from chronos_esm import main, io as model_io, data, log_utils, config
import jax.numpy as jnp

# Load 10-Year State
if not Path("{RESUME_FILE}").exists():
    print("Error: Resume file not found!")
    sys.exit(1)
    
print("Loading resume state from {RESUME_FILE}...")
state = model_io.load_state_from_netcdf("{RESUME_FILE}")

# Calculate Steps for remaining 90 Years
years_to_run = 90.0
dt_coupled = 900.0 # 15 mins
dt_atmos = 150.0   # 5x Speedup (Safe)

total_steps = int(years_to_run * 365 * 24 * 3600 / dt_coupled)
steps_per_year = int(365 * 24 * 3600 / dt_coupled)

print(f"Running {{years_to_run}} years ({{total_steps}} steps)...")

params = main.ModelParams(co2_ppm=280.0, solar_constant=1361.0, mask=data.load_bathymetry_mask(nz=15))
reg = main.regrid.Regridder()
PHYS_PARAMS = {{"qc_ref": 0.4915, "epsilon_smooth": 4.71e-2}}

output_dir = Path("{OUTPUT_DIR}")
logger = log_utils.setup_logger("century", output_dir / "model.log")

start_time = time.time()

for step in range(1, total_steps + 1):
    # Stable Physics Settings
    state = main.step_coupled(
        state, params, reg, 
        physics_params=PHYS_PARAMS,
        Ah=5.0e5,  
        shapiro_strength=0.2, 
        kappa_gm=2000.0 
    )
    
    # Stability Check
    if step % 500 == 0:
        elapsed = time.time() - start_time
        rate = step / elapsed
        state.ocean.temp.block_until_ready()
        print(f"Step {{step}}/{{total_steps}} ({{rate:.2f}} steps/s)")
        
        t_global = float(state.atmos.temp.mean())
        if jnp.isnan(t_global):
             logger.error("NaN Detected!")
             sys.exit(1)

    # Save Yearly
    if step % steps_per_year == 0:
        year_idx = 10 + (step // steps_per_year) # Start from Year 11
        fname = output_dir / f"year_{{year_idx:03d}}.nc"
        model_io.save_state_to_netcdf(state, fname)
        t_mean = float(state.atmos.temp.mean())
        logger.info(f"Year {{year_idx}}: Global T={{t_mean:.2f}}")

print("Century Run Complete.")
model_io.save_state_to_netcdf(state, output_dir / "final_state_100y.nc")
"""

    runner_script = OUTPUT_DIR / "run_core.py"
    with open(runner_script, "w") as f:
        f.write(script_content)
        
    env = os.environ.copy()
    env["CHRONOS_DT_ATMOS"] = "150.0" 
    
    subprocess.run([sys.executable, str(runner_script)], env=env, check=True)

if __name__ == "__main__":
    run_century()
