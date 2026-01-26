
import sys
import os
import subprocess
import time
from pathlib import Path
import jax.numpy as jnp

# Outputs
OUTPUT_DIR = Path("outputs/long_run_fast")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_phase_1():
    """Run 1 month stabilization at DT=30s"""
    print("=== PHASE 1: STABILIZATION (DT=30s) ===")
    
    script_content = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from chronos_esm import main, io as model_io, data
import jax.numpy as jnp

# Load Restart
prod_dir = Path("outputs/production_control")
restarts = sorted(list(prod_dir.glob("restart_*.nc")))
if not restarts:
    print("No restart found!")
    sys.exit(1)
latest = restarts[-1]
state = model_io.load_state_from_netcdf(latest)

# Run 1 Month
years = 1.0/12.0
dt = 900.0
steps = int((years * 365 * 24 * 3600) / dt) + 1 

print(f"Running Phase 1 for {steps} steps (DT_ATMOS=30s)...")

params = main.ModelParams(co2_ppm=280.0, solar_constant=1361.0, mask=data.load_bathymetry_mask(nz=15))
reg = main.regrid.Regridder()
PHYS_PARAMS = {"qc_ref": 0.4915, "epsilon_smooth": 4.71e-2}

# Use Strong Diffusion here too just in case
for i in range(steps):
    state = main.step_coupled(
        state, params, reg, 
        physics_params=PHYS_PARAMS,
        Ah=5.0e5, shapiro_strength=0.1
    )
    if i % 100 == 0:
        print(f"Phase 1 Step {i}")
        state.ocean.temp.block_until_ready()

model_io.save_state_to_netcdf(state, "outputs/long_run_fast/stable_restart.nc")
print("Phase 1 Complete.")
"""
    p1_script = OUTPUT_DIR / "phase1.py"
    with open(p1_script, "w") as f:
        f.write(script_content)
        
    env = os.environ.copy()
    env["CHRONOS_DT_ATMOS"] = "30.0" # Explicit safety
    subprocess.run([sys.executable, str(p1_script)], env=env, check=True)


def run_phase_2(years=10):
    """Run N years at High Speed (DT=150s)"""
    print(f"=== PHASE 2: HIGH SPEED RUN ({years} Years, DT=150s) ===")
    
    script_content = f"""
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from chronos_esm import main, io as model_io, data, log_utils, config
import jax.numpy as jnp

# Load Phase 1 Restart
state = model_io.load_state_from_netcdf("outputs/long_run_fast/stable_restart.nc")

years = {years}
dt_coupled = 900.0
total_steps = int(years * 365 * 24 * 3600 / dt_coupled)
steps_per_year = int(365 * 24 * 3600 / dt_coupled)

print(f"Running {{total_steps}} coupled steps...")
print(f"Config DT_ATMOS Check: {{config.DT_ATMOS}}")

params = main.ModelParams(co2_ppm=280.0, solar_constant=1361.0, mask=data.load_bathymetry_mask(nz=15))
reg = main.regrid.Regridder()
PHYS_PARAMS = {{"qc_ref": 0.4915, "epsilon_smooth": 4.71e-2}}

output_dir = Path("outputs/long_run_fast")
logger = log_utils.setup_logger("phase2", output_dir / "model.log")

start_time = time.time()

for step in range(total_steps):
    # CRITICAL: High Diffusion for Stability with large DT
    state = main.step_coupled(
        state, params, reg, 
        physics_params=PHYS_PARAMS,
        Ah=5.0e5,  # Strong Laplacian Viscosity
        shapiro_strength=0.2, # Strong Filter
        kappa_gm=2000.0 # Enhanced eddy mixing
    )
    
    # Stability Check every 100 steps
    if step % 100 == 0:
        t_global = float(state.atmos.temp.mean())
        if jnp.isnan(t_global) or t_global > 350 or t_global < 150:
            logger.error(f"Instability detected at step {{step}}! T={{t_global}}")
            sys.exit("Model Diverged")
            
    if step % 500 == 0:
        elapsed = time.time() - start_time
        rate = (step+1) / elapsed
        print(f"Step {{step}}/{{total_steps}} ({{rate:.2f}} steps/s)")
    
    if step > 0 and step % steps_per_year == 0:
        year = step // steps_per_year
        fname = output_dir / f"year_{{year:03d}}.nc"
        model_io.save_state_to_netcdf(state, fname)
        t_mean = float(state.atmos.temp.mean())
        logger.info(f"Year {{year}}: Global T={{t_mean:.2f}}")

print("Phase 2 Complete.")
model_io.save_state_to_netcdf(state, output_dir / "final_state.nc")
"""

    p2_script = OUTPUT_DIR / "phase2.py"
    with open(p2_script, "w") as f:
        f.write(script_content)
        
    env = os.environ.copy()
    env["CHRONOS_DT_ATMOS"] = "150.0"  # Reduced from 225s for safety
    
    subprocess.run([sys.executable, str(p2_script)], env=env, check=True)

if __name__ == "__main__":
    run_phase_1()
    run_phase_2(years=10)
