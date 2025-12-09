"""
Resume Control Run.

Loads the latest state_XXXX.nc file and continues the simulation.
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
from chronos_esm import logging as c_log  # noqa: E402
from chronos_esm.atmos import qtcm  # noqa: E402
from chronos_esm.coupler import state as coupled_state  # noqa: E402
from chronos_esm.ice import driver as ice_driver  # noqa: E402
from chronos_esm.land import driver as land_driver  # noqa: E402
from chronos_esm.ocean import veros_driver  # noqa: E402


def load_latest_state(output_dir: Path):
    """Load the latest NetCDF state."""
    files = sorted(glob.glob(str(output_dir / "state_*.nc")))
    if not files:
        raise FileNotFoundError(f"No state files found in {output_dir}")

    latest_file = files[-1]
    print(f"Loading state from {latest_file}...")

    ds = xr.open_dataset(latest_file)

    # Reconstruct State
    # Ocean
    ocean = veros_driver.OceanState(
        u=jnp.array(ds.ocean_u.values),
        v=jnp.array(ds.ocean_v.values),
        w=jnp.zeros_like(
            jnp.array(ds.ocean_u.values)
        ),  # w usually diagnostic or not saved?
        temp=jnp.array(ds.ocean_temp.values),
        salt=jnp.array(ds.ocean_salt.values),
        psi=jnp.zeros((ds.ocean_u.shape[1], ds.ocean_u.shape[2])),  # Re-solve psi
        rho=jnp.zeros_like(jnp.array(ds.ocean_temp.values)),  # Re-calc rho
    )

    # Atmos
    atmos = qtcm.AtmosState(
        u=jnp.array(ds.atmos_u.values),
        v=jnp.array(ds.atmos_v.values),
        temp=jnp.array(ds.atmos_temp.values),
        q=jnp.array(ds.atmos_q.values),
    )

    # Ice
    ice = ice_driver.IceState(
        thickness=jnp.array(ds.ice_thickness.values),
        concentration=jnp.array(ds.ice_concentration.values),
        surface_temp=jnp.ones_like(jnp.array(ds.ice_thickness.values))
        * -1.8,  # Approx or save?
    )

    # Land (Not saved in current IO! Need to init or save)
    # TODO: Update IO to save Land. For now, re-init Land (bucket empty/half?)
    # This is a limitation. Let's re-init land for now.
    ny, nx = ds.atmos_temp.shape
    land = land_driver.init_land_state(ny, nx)

    # Fluxes (Init zero/SST)
    fluxes = coupled_state.FluxState(
        net_heat_flux=jnp.zeros((ny, nx)),
        freshwater_flux=jnp.zeros((ny, nx)),
        wind_stress_x=jnp.zeros((ny, nx)),
        wind_stress_y=jnp.zeros((ny, nx)),
        sst=jnp.array(
            ds.atmos_temp.values
        ),  # Approx SST from Atmos T? Or use Ocean Surface T
    )

    # Time
    time_val = float(ds.time.values)

    state = coupled_state.CoupledState(
        ocean=ocean, atmos=atmos, ice=ice, land=land, fluxes=fluxes, time=time_val
    )

    # Extract month index from filename
    month_idx = int(Path(latest_file).stem.split("_")[1])

    return state, month_idx


def resume_run(target_years: float = 100.0):
    """Resume simulation."""
    output_dir = Path("outputs/control_run")

    # Load State
    try:
        current_state, start_month = load_latest_state(output_dir)
    except Exception as e:
        print(f"Failed to load state: {e}")
        return

    # Setup Logger
    logger = c_log.setup_logger("resume_run", output_dir / "resume.log")
    logger.info(f"Resuming run from Month {start_month}...")

    # Config
    seconds_per_year = 365 * 24 * 3600
    seconds_per_month = seconds_per_year / 12.0
    dt = 450.0  # Updated DT_ATMOS

    steps_per_month = int(seconds_per_month / dt)
    n_months = int(target_years * 12)

    regridder = main.regrid.Regridder()
    mask = data.load_bathymetry_mask()
    params = main.ModelParams(mask=mask)

    # Compile Step
    def month_step_fn(carry, _):
        s = carry
        new_s = main.step_coupled(s, params, regridder)
        return new_s, None

    @jax.jit
    def run_one_month(start_state):
        final_s, _ = jax.lax.scan(
            month_step_fn, start_state, jnp.arange(steps_per_month)
        )
        return final_s

    logger.info("Starting simulation loop...")

    for month in range(start_month + 1, n_months + 1):
        logger.info(f"Running Month {month}/{n_months}...")
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
            f"Month {month} complete in {t_end - t_start:.2f}s. T_mean={t_mean:.2f}K. Saving output..."
        )
        model_io.save_state_to_netcdf(
            current_state, output_dir / f"state_{month:04d}.nc"
        )


if __name__ == "__main__":
    resume_run(years=100.0)
