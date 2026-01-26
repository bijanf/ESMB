"""
Run Scenario Experiment.

Loads a restart file (e.g. from control run spin-up) and continues with a specific scenario
(e.g., 1% CO2 increase).
"""

import glob
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import xarray as xr

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import data  # noqa: E402
from chronos_esm import main  # noqa: E402
from chronos_esm import io as model_io  # noqa: E402
from chronos_esm import log_utils as c_log  # noqa: E402
from chronos_esm.atmos import dynamics as atmos_driver  # noqa: E402
from chronos_esm.coupler import state as coupled_state  # noqa: E402
from chronos_esm.ice import driver as ice_driver  # noqa: E402
from chronos_esm.land import driver as land_driver  # noqa: E402
from chronos_esm.ocean import veros_driver  # noqa: E402
from chronos_esm.config import DT_ATMOS, NZ_OCEAN


def load_restart_state(restart_file: Path):
    """Load a specific restart NetCDF state."""
    print(f"Loading state from {restart_file}...")

    ds = xr.open_dataset(restart_file)

    # Reconstruct State
    # Ocean
    # Assuming restart file has all prognostic variables
    ocean = veros_driver.OceanState(
        u=jnp.array(ds.ocean_u.values),
        v=jnp.array(ds.ocean_v.values),
        w=jnp.array(ds.ocean_w.values),
        temp=jnp.array(ds.ocean_temp.values),
        salt=jnp.array(ds.ocean_salt.values),
        psi=jnp.array(ds.ocean_psi.values) if 'ocean_psi' in ds else jnp.zeros_like(ds.ocean_u.values[0]),
        rho=jnp.zeros_like(jnp.array(ds.ocean_temp.values)),  # Re-calc rho in first step
        dic=jnp.array(ds.ocean_dic.values),
    )

    # Atmos
    # Need to match AtmosState fields
    atmos = atmos_driver.AtmosState(
        vorticity=jnp.array(ds.atmos_vorticity.values),
        divergence=jnp.array(ds.atmos_divergence.values),
        temp=jnp.array(ds.atmos_temp.values),
        ln_ps=jnp.array(ds.atmos_ln_ps.values),
        q=jnp.array(ds.atmos_q.values),
        co2=jnp.array(ds.atmos_co2.values),
        u=jnp.array(ds.atmos_u.values),
        v=jnp.array(ds.atmos_v.values),
        psi=jnp.zeros_like(ds.atmos_u.values), # diagnostic
        chi=jnp.zeros_like(ds.atmos_u.values), # diagnostic
    )

    # Ice
    ice = ice_driver.IceState(
        thickness=jnp.array(ds.ice_thickness.values),
        concentration=jnp.array(ds.ice_concentration.values),
        surface_temp=jnp.array(ds.ice_surface_temp.values) if 'ice_surface_temp' in ds else -1.8 * jnp.ones_like(ds.ice_thickness.values),
    )

    # Land
    # Re-init land if not in restart (often simplified)
    ny, nx = ds.atmos_temp.shape
    # Try to load if available, else init
    # Assuming Land state is simple (temp, soil_moisture, snow)
    # Check if variables exist in ds
    if 'land_temp' in ds:
         land = land_driver.LandState(
             temp=jnp.array(ds.land_temp.values),
             soil_moisture=jnp.array(ds.land_soil_moisture.values),
             snow_depth=jnp.array(ds.land_snow_depth.values),
             lai=jnp.array(ds.land_lai.values) if 'land_lai' in ds else 2.0 * jnp.ones((ny, nx)),
         )
    else:
         print("Warning: initializing fresh Land state (not found in restart)")
         land = land_driver.init_land_state(ny, nx)

    # Fluxes
    fluxes = coupled_state.FluxState(
        net_heat_flux=jnp.array(ds.flux_net_heat.values) if 'flux_net_heat' in ds else jnp.zeros((ny, nx)),
        freshwater_flux=jnp.array(ds.flux_freshwater.values) if 'flux_freshwater' in ds else jnp.zeros((ny, nx)),
        wind_stress_x=jnp.array(ds.flux_tau_x.values) if 'flux_tau_x' in ds else jnp.zeros((ny, nx)),
        wind_stress_y=jnp.array(ds.flux_tau_y.values) if 'flux_tau_y' in ds else jnp.zeros((ny, nx)),
        precip=jnp.array(ds.flux_precip.values) if 'flux_precip' in ds else jnp.zeros((ny, nx)),
        sst=jnp.array(ds.flux_sst.values) if 'flux_sst' in ds else jnp.zeros((ny, nx)),
        carbon_flux_ocean=jnp.zeros((ny, nx)),
        carbon_flux_land=jnp.zeros((ny, nx)),
    )

    # Time
    time_val = float(ds.time.values) if 'time' in ds else 0.0

    state = coupled_state.CoupledState(
        ocean=ocean, atmos=atmos, ice=ice, land=land, fluxes=fluxes, time=time_val
    )
    
    return state


def run_scenario(restart_path: str, co2_rate: float, years: float = 50.0):
    """Run a scenario simulation starting from restart."""
    restart_file = Path(restart_path)
    if not restart_file.exists():
        print(f"Error: Restart file {restart_file} not found.")
        return

    output_dir = Path(f"outputs/scenario_co2_{int(co2_rate*100)}pct")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = c_log.setup_logger("scenario_run", output_dir / "model.log")
    logger.info(f"Starting Scenario Run (CO2 +{co2_rate*100}%/yr) for {years} years...")
    logger.info(f"Restarting from: {restart_file}")

    # Load State
    try:
        current_state = load_restart_state(restart_file)
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return

    # Extract start month from time or filename
    # Time is in seconds.
    seconds_per_year = 365 * 24 * 3600
    start_time = current_state.time
    start_month = int(start_time / (seconds_per_year / 12.0))
    logger.info(f"Simulation time: {start_time:.2e} s (Month {start_month})")

    # Config
    dt = DT_ATMOS
    seconds_per_month = seconds_per_year / 12.0
    steps_per_month = int(seconds_per_month / dt)
    n_months = int(years * 12)

    logger.info(f"Running for {n_months} additional months...")

    regridder = main.regrid.Regridder()
    mask = data.load_bathymetry_mask()
    
    # scenario params
    params = main.ModelParams(mask=mask, co2_increase_rate=co2_rate)

    @jax.jit
    def run_one_month(start_state):
        def month_step_fn(carry, _):
             s = carry
             new_s = main.step_coupled(s, params, regridder)
             return new_s, None

        final_s, _ = jax.lax.scan(
            month_step_fn, start_state, jnp.arange(steps_per_month)
        )
        return final_s

    logger.info("Starting simulation loop...")

    for i in range(1, n_months + 1):
        month_idx = start_month + i
        logger.info(f"Running Month {month_idx} (Scenario Month {i})...")
        t_start = time.time()

        current_state = run_one_month(current_state)
        current_state.ocean.temp.block_until_ready()
        t_end = time.time()

        # Stability Check
        t_mean = float(jnp.mean(current_state.atmos.temp))
        if jnp.isnan(t_mean) or t_mean > 350.0 or t_mean < 150.0:
            logger.error(f"Stability Check Failed! Global Mean Temp: {t_mean} K")
            break

        logger.info(
            f"Month {month_idx} complete in {t_end - t_start:.2f}s. T_mean={t_mean:.2f}K."
        )

        # Save output
        model_io.save_state_to_netcdf(
            current_state, output_dir / f"mean_{month_idx:04d}.nc"
        )
        
        # Checkpoint every 10 years
        if month_idx % 120 == 0:
             chk_name = output_dir / f"restart_{month_idx:04d}.nc"
             model_io.save_state_to_netcdf(current_state, chk_name)
             logger.info(f"Saved Checkpoint: {chk_name}")

    logger.info("Scenario Run Complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python experiments/run_scenario.py <restart_file> [co2_rate]")
        sys.exit(1)
        
    restart_path = sys.argv[1]
    co2_rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
    
    run_scenario(restart_path, co2_rate, years=50.0)
