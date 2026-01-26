"""
Production Control Run Experiment.

Runs the coupled model with WOA18 initial conditions for 100 years.
Outputs monthly means for analysis (AMOC, GMT).
"""

import sys
import time
from pathlib import Path
from typing import List, NamedTuple, Tuple, Union  # noqa: F401

import jax
import jax.numpy as jnp  # noqa: F401

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import data  # noqa: E402

from chronos_esm import main  # noqa: E402
from chronos_esm import io as model_io  # noqa: E402
from chronos_esm.atmos import dynamics as atmos_driver  # noqa: E402
from chronos_esm.coupler import state as coupled_state  # noqa: E402
from chronos_esm import data  # noqa: E402

# from chronos_esm.ice import driver as ice_driver  # noqa: E402 # Unused
from chronos_esm.land import driver as land_driver  # noqa: E402
from chronos_esm.ocean import veros_driver  # noqa: E402


def setup_control_run():
    """Initialize model with real ICs."""
    print("Loading initial conditions (WOA18)...")
    temp_ic, salt_ic = data.load_initial_conditions(nz=15)

    # Create Ocean State with loaded ICs
    # Velocities zero
    ny, nx = temp_ic.shape[1], temp_ic.shape[2]
    # ny, nx = 96, 192
    nz = temp_ic.shape[0]

    ocean = veros_driver.OceanState(
        u=jnp.zeros((nz, ny, nx)),
        v=jnp.zeros((nz, ny, nx)),
        w=jnp.zeros((nz, ny, nx)),
        temp=temp_ic + 273.15,  # C -> K
        salt=salt_ic,
        psi=jnp.zeros((ny, nx)),
        rho=jnp.zeros((nz, ny, nx)),  # Will be updated in first step
        dic=jnp.ones((nz, ny, nx)) * 2000.0,  # Initial DIC
    )

    # Initialize Atmos
    # Use dynamics initialization (Jablonowski-Williamson)
    atmos = atmos_driver.init_atmos_state(ny, nx)

    # We could overwrite temp with SST-based guess, but dynamics init is balanced.
    # Let's keep dynamics init.
    sst_ic = temp_ic[0] + 273.15

    # Initialize Land

    land = land_driver.init_land_state(ny, nx)

    # Coupled State
    state = coupled_state.init_coupled_state(ocean, atmos, land)

    # Update fluxes SST to match
    fluxes = state.fluxes._replace(sst=sst_ic)
    state = state._replace(fluxes=fluxes)

    return state


def run_control(years: float = 100.0):
    """Run control simulation."""
    state = setup_control_run()

    # Calculate steps
    # DT_ATMOS is usually smaller (e.g. 1800s).
    # 1 year = 365 * 24 * 3600 seconds
    seconds_per_year = 365 * 24 * 3600
    seconds_per_month = seconds_per_year / 12.0
    from chronos_esm.config import DT_ATMOS, DT_OCEAN, GRAVITY, OMEGA, P0  # noqa: F401

    if "dt" in locals():
         del dt # clear previous definition if any
         
    # Control run calculates steps based on the coupled step size.
    # step_coupled now advances by DT_OCEAN (synchronizing multiple atmos steps inside).
    dt = DT_OCEAN

    steps_per_month = int(seconds_per_month / dt)
    n_months = int(years * 12)

    # Setup Logger
    from chronos_esm import log_utils as c_log

    output_dir = Path("outputs/production_control")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = c_log.setup_logger("production_control", output_dir / "model.log")

    logger.info(f"Starting Production Control Run for {years} years ({n_months} months)...")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Ocean Timestep: {DT_OCEAN}s, Atmos Timestep: {DT_ATMOS}s")

    # Save initial state
    logger.info("Saving initial state (state_0000.nc)...")
    model_io.save_state_to_netcdf(state, output_dir / "state_0000.nc")

    t0 = time.time()

    regridder = main.regrid.Regridder()

    # Load Mask
    logger.info("Loading Land Mask...")
    mask = data.load_bathymetry_mask(nz=15) # Default match
    
    # Solar Constant 1361.0 (Physical Default)
    # CO2 = 280 ppm (Pre-industrial)
    params = main.ModelParams(
        co2_ppm=280.0,
        solar_constant=1361.0,
        mask=mask
    )

    # Compile the step function (scan over 1 month)
    # We accumulate fields for monthly means
    
    class Accumulator(NamedTuple):
        # Atmos
        atmos_temp: jnp.ndarray
        atmos_u: jnp.ndarray
        atmos_v: jnp.ndarray
        atmos_q: jnp.ndarray
        atmos_co2: jnp.ndarray
        atmos_vorticity: jnp.ndarray
        atmos_divergence: jnp.ndarray
        atmos_ln_ps: jnp.ndarray
        # Ocean
        ocean_temp: jnp.ndarray
        ocean_salt: jnp.ndarray
        ocean_u: jnp.ndarray
        ocean_v: jnp.ndarray
        ocean_w: jnp.ndarray
        ocean_psi: jnp.ndarray
        ocean_dic: jnp.ndarray
        # Fluxes
        sst: jnp.ndarray
        precip: jnp.ndarray
        net_heat_flux: jnp.ndarray
        wind_stress_x: jnp.ndarray
        wind_stress_y: jnp.ndarray
        # Land
        land_temp: jnp.ndarray
        land_snow: jnp.ndarray

    def extract_fields(s):
        return Accumulator(
            atmos_temp=s.atmos.temp,
            atmos_u=s.atmos.u,
            atmos_v=s.atmos.v,
            atmos_q=s.atmos.q,
            atmos_co2=s.atmos.co2,
            atmos_vorticity=s.atmos.vorticity,
            atmos_divergence=s.atmos.divergence,
            atmos_ln_ps=s.atmos.ln_ps,
            ocean_temp=s.ocean.temp,
            ocean_salt=s.ocean.salt,
            ocean_u=s.ocean.u,
            ocean_v=s.ocean.v,
            ocean_w=s.ocean.w,
            ocean_psi=s.ocean.psi, # Diagnostic streamfunction (might be zero)
            ocean_dic=s.ocean.dic,
            sst=s.fluxes.sst,
            precip=s.fluxes.precip,
            net_heat_flux=s.fluxes.net_heat_flux,
            wind_stress_x=s.fluxes.wind_stress_x,
            wind_stress_y=s.fluxes.wind_stress_y,
            land_temp=s.land.temp,
            land_snow=s.land.snow_depth,
        )

    def zero_accumulator(s):
        fields = extract_fields(s)
        return jax.tree_util.tree_map(jnp.zeros_like, fields)

    def month_step_fn(carry, _):
        s, acc = carry
        # Use strong viscosity/diffusion to suppress grid-scale noise (T31)
        new_s = main.step_coupled(
            s, params, regridder,
            r_drag=5.0e-2,
            kappa_gm=1000.0,
            kappa_h=1000.0,
            kappa_bi=0.0,
            Ah=2.0e5,  # Strong Laplacian Viscosity
            Ab=0.0,
            shapiro_strength=0.1 # Strong Shapiro Filter
        )

        # Accumulate
        current_fields = extract_fields(new_s)
        new_acc = jax.tree_util.tree_map(lambda x, y: x + y, acc, current_fields)

        return (new_s, new_acc), None

    # JIT compile the monthly loop
    @jax.jit
    def run_one_month(start_state):
        # Initialize accumulator with zeros
        init_acc = zero_accumulator(start_state)

        # Scan
        (final_s, final_acc), _ = jax.lax.scan(
            month_step_fn, (start_state, init_acc), jnp.arange(steps_per_month)
        )

        return final_s, final_acc

    # Restart Logic
    restart_files = sorted(list(output_dir.glob("restart_*.nc")))
    start_month = 1
    current_state = state

    if restart_files:
        latest_restart = restart_files[-1]
        logger.info(f"Found restart file: {latest_restart}")
        try:
            # Extract month from filename "restart_{month:04d}.nc"
            # Filename is the last part of the path
            restart_month_str = latest_restart.name.split("_")[1].split(".")[0]
            start_month = int(restart_month_str) + 1
            
            logger.info(f"Resuming from Month {start_month}...")
            
            # Load the state
            current_state = model_io.load_state_from_netcdf(latest_restart)
            
            # Ensure we have the correct time stepping or other params if they changed
            # But state loading should handle data.
        except Exception as e:
            logger.error(f"Failed to load restart file {latest_restart}: {e}")
            logger.info("Starting from scratch...")
            start_month = 1
            current_state = state

    logger.info("Starting simulation loop...")

    for month in range(start_month, n_months + 1):
        logger.info(f"Running Month {month}/{n_months}...")
        t_start = time.time()

        current_state, accumulated = run_one_month(current_state)

        # Block to ensure completion and timing
        current_state.ocean.temp.block_until_ready()
        t_end = time.time()

        # Stability Check
        t_mean = float(jnp.mean(current_state.atmos.temp))
        if jnp.isnan(t_mean) or t_mean > 350.0 or t_mean < 150.0:
            logger.error(f"Stability Check Failed! Global Mean Temp: {t_mean} K")
            logger.error("Model has diverged. Stopping simulation.")
            break

        logger.info(
            f"Month {month} complete in {t_end - t_start:.2f}s. T_mean={t_mean:.2f}K."
        )

        # Compute Means (divide by steps)
        means = jax.tree_util.tree_map(lambda x: x / steps_per_month, accumulated)

        # Save Monthly Mean
        mean_filename = output_dir / f"mean_{month:04d}.nc"

        mean_ocean = current_state.ocean._replace(
            temp=means.ocean_temp,
            salt=means.ocean_salt,
            u=means.ocean_u,
            v=means.ocean_v,
            w=means.ocean_w,
            psi=means.ocean_psi,
            dic=means.ocean_dic,
        )
        mean_atmos = current_state.atmos._replace(
            temp=means.atmos_temp,
            u=means.atmos_u,
            v=means.atmos_v,
            q=means.atmos_q,
            co2=means.atmos_co2,
            vorticity=means.atmos_vorticity,
            divergence=means.atmos_divergence,
            ln_ps=means.atmos_ln_ps,
        )
        mean_fluxes = current_state.fluxes._replace(
            sst=means.sst, precip=means.precip, net_heat_flux=means.net_heat_flux,
            wind_stress_x=means.wind_stress_x, wind_stress_y=means.wind_stress_y
        )
        mean_land = current_state.land._replace(
            temp=means.land_temp,
            snow_depth=means.land_snow
        )

        mean_state_obj = current_state._replace(
            ocean=mean_ocean, atmos=mean_atmos, fluxes=mean_fluxes, land=mean_land
        )

        model_io.save_state_to_netcdf(mean_state_obj, mean_filename)

        # Checkpoint (Restart) every 1 year (12 months)
        if month % 12 == 0:
            logger.info(f"Saving Restart Checkpoint at Month {month}...")
            restart_filename = output_dir / f"restart_{month:04d}.nc"
            model_io.save_state_to_netcdf(current_state, restart_filename)

    t1 = time.time()

    logger.info(f"Run complete in {t1-t0:.2f}s")
    logger.info(f"Final Global Mean Temp: {jnp.mean(current_state.atmos.temp):.2f} K")

    return current_state


if __name__ == "__main__":
    # Run for 100 years
    run_control(years=100.0)
