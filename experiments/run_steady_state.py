
import sys
import time
import shutil
from pathlib import Path
from typing import NamedTuple
import jax.numpy as jnp
import jax

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import main, io as model_io, data, log_utils as c_log
from chronos_esm.config import DT_ATMOS, DT_OCEAN

# Tuned Parameters
PHYSICS_PARAMS = {
    "qc_ref": 0.4915,
    "epsilon_smooth": 4.71e-2
}

OUTPUT_DIR = Path("outputs/steady_state_test")

def setup_simulation():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for restart file from production
    prod_dir = Path("outputs/production_control")
    restarts = sorted(list(prod_dir.glob("restart_*.nc")))
    
    if restarts:
        latest = restarts[-1]
        print(f"Initializing from {latest}...")
        state = model_io.load_state_from_netcdf(latest)
    else:
        print("No restart found. Initializing from scratch (WOA18)...")
        # Reuse production setup logic if needed, but for now assume restart exists
        # as user just ran verification on it.
        sys.exit("Error: No restart file found in outputs/production_control. Run failed.")
        
    return state

def run_steady_state(years=1.0):
    state = setup_simulation()
    
    # Calculate steps
    seconds_per_year = 365 * 24 * 3600
    seconds_per_month = seconds_per_year / 12.0
    dt_coupled = DT_OCEAN
    steps_per_month = int(seconds_per_month / dt_coupled)
    n_months = int(years * 12)
    
    logger = c_log.setup_logger("steady_state", OUTPUT_DIR / "model.log")
    
    print(f"Running for {years} years ({n_months} months)...")
    print(f"Steps per month: {steps_per_month}")
    
    regridder = main.regrid.Regridder()
    
    # Params
    params = main.ModelParams(
        co2_ppm=280.0,
        solar_constant=1361.0,
        mask=data.load_bathymetry_mask(nz=15)
    )
    
    # Accumulator (Same as production)
    class Accumulator(NamedTuple):
        atmos_temp: jnp.ndarray
        atmos_q: jnp.ndarray
        atmos_ln_ps: jnp.ndarray
        sst: jnp.ndarray
        precip: jnp.ndarray
        
    def extract_fields(s):
        return Accumulator(
            atmos_temp=s.atmos.temp,
            atmos_q=s.atmos.q,
            atmos_ln_ps=s.atmos.ln_ps,
            sst=s.fluxes.sst,
            precip=s.fluxes.precip
        )
        
    def zero_accumulator(s):
        fields = extract_fields(s)
        return jax.tree_util.tree_map(jnp.zeros_like, fields)

    def month_step_fn(carry, _):
        s, acc = carry
        # Use production settings + TUNED PARAMETERS
        new_s = main.step_coupled(
            s, params, regridder,
            r_drag=5.0e-2,
            kappa_gm=1000.0,
            kappa_h=1000.0,
            Ah=2.0e5,
            shapiro_strength=0.1,
            physics_params=PHYSICS_PARAMS # Pass Tuned Params
        )
        
        current_fields = extract_fields(new_s)
        new_acc = jax.tree_util.tree_map(lambda x, y: x + y, acc, current_fields)
        return (new_s, new_acc), None

    @jax.jit
    def run_one_month(start_state):
        init_acc = zero_accumulator(start_state)
        (final_s, final_acc), _ = jax.lax.scan(
            month_step_fn, (start_state, init_acc), jnp.arange(steps_per_month)
        )
        return final_s, final_acc

    current_state = state
    
    for month in range(1, n_months + 1):
        t0 = time.time()
        current_state, accumulated = run_one_month(current_state)
        current_state.ocean.temp.block_until_ready()
        t1 = time.time()
        
        t_mean = float(jnp.mean(current_state.atmos.temp))
        logger.info(f"Month {month}: T_mean={t_mean:.2f}K. Time={t1-t0:.2f}s")
        
        # Save Monthly Mean (Subset)
        means = jax.tree_util.tree_map(lambda x: x / steps_per_month, accumulated)
        
        # We need to construct a full state to save using io.save_state_to_netcdf
        # Or just save specific fields using custom xarray logic
        # For simplicity, let's reuse io.save_state but update key fields
        # Note: This is hacky, better to use full State object like production logic.
        # But we only need T, Precip, P for diagnosis.
        
        # Let's just update the main state with mean values for ATMOS fields and save that
        
        mean_atmos = current_state.atmos._replace(
            temp=means.atmos_temp,
            q=means.atmos_q,
            ln_ps=means.atmos_ln_ps
        )
        mean_fluxes = current_state.fluxes._replace(
            sst=means.sst,
            precip=means.precip
        )
        
        # Keep ocean instantaneous for now (or means if we implemented full accumulator)
        mean_state = current_state._replace(atmos=mean_atmos, fluxes=mean_fluxes)
        
        model_io.save_state_to_netcdf(mean_state, OUTPUT_DIR / f"mean_{month:04d}.nc")
        
    print("Simulation Complete.")

if __name__ == "__main__":
    run_steady_state()
