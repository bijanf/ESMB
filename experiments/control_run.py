"""
Control Run Experiment.

Runs the coupled model with WOA18 initial conditions for a specified duration.
"""

import sys
import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import data  # noqa: E402
from chronos_esm import main  # noqa: E402
from chronos_esm import io as model_io  # noqa: E402
from chronos_esm.atmos import dynamics as atmos_driver  # noqa: E402
from chronos_esm.coupler import state as coupled_state  # noqa: E402

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


def run_control(years: float = 1.0):
    """Run control simulation."""
    state = setup_control_run()

    # Calculate steps
    # DT_ATMOS is usually smaller (e.g. 1800s).
    # 1 year = 365 * 24 * 3600 seconds
    seconds_per_year = 365 * 24 * 3600
    seconds_per_month = seconds_per_year / 12.0
    from chronos_esm.config import DT_ATMOS

    dt = DT_ATMOS

    steps_per_month = int(seconds_per_month / dt)
    n_months = int(years * 12)

    # Setup Logger
    from chronos_esm import log_utils as c_log

    output_dir = Path("outputs/control_run")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = c_log.setup_logger("control_run", output_dir / "model.log")

    logger.info(f"Starting Control Run for {years} years ({n_months} months)...")
    logger.info(f"Output directory: {output_dir}")

    # Save initial state
    logger.info("Saving initial state (state_0000.nc)...")
    model_io.save_state_to_netcdf(state, output_dir / "state_0000.nc")

    t0 = time.time()

    regridder = main.regrid.Regridder()

    # Load Mask
    logger.info("Loading Land Mask...")
    mask = data.load_bathymetry_mask()
    logger.info(f"Ocean Fraction: {jnp.mean(mask):.2f}")

    params = main.ModelParams(mask=mask)

    # Compile the step function (scan over 1 month)
    # We accumulate fields for monthly means
    # Define Accumulator structure
    # (using a simple tuple for now to avoid JIT issues with custom classes if
    # not careful, but NamedTuple is fine in JAX)

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
            ocean_psi=s.ocean.psi,
            ocean_dic=s.ocean.dic,
            sst=s.fluxes.sst,
            precip=s.fluxes.precip,
            net_heat_flux=s.fluxes.net_heat_flux,
        )

    def zero_accumulator(s):
        fields = extract_fields(s)
        return jax.tree_util.tree_map(jnp.zeros_like, fields)

    def month_step_fn(carry, _):
        s, acc = carry
        new_s = main.step_coupled(s, params, regridder)

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

    current_state = state

    logger.info("Starting simulation loop...")

    for month in range(1, n_months + 1):
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
        # We do this on host or device? Device is better.
        # But we need to save them.
        means = jax.tree_util.tree_map(lambda x: x / steps_per_month, accumulated)

        # Save Monthly Mean
        # We need to package this into a structure that save_state_to_netcdf
        # can handle, OR we create a custom saver for means.
        # OR we create a custom saver for means.
        # For simplicity, let's create a "MeanState" object that looks like CoupledState but with mean values.
        # However, CoupledState requires specific types.
        # Easier to just save the dictionary of means.

        # Convert to numpy for saving
        # means_np = jax.tree_util.tree_map(lambda x: np.array(x), means)

        # Save means
        mean_filename = output_dir / f"mean_{month:04d}.nc"
        # We can use a custom save function or reuse existing if we reconstruct state
        # Reconstructing state is cleaner for standard tools

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
            sst=means.sst, precip=means.precip, net_heat_flux=means.net_heat_flux
        )

        mean_state_obj = current_state._replace(
            ocean=mean_ocean, atmos=mean_atmos, fluxes=mean_fluxes
        )

        model_io.save_state_to_netcdf(mean_state_obj, mean_filename)

        # Checkpoint (Restart) every 10 years (120 months)
        if month % 120 == 0:
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
    # Short verification run
    # run_control(years=0.01) # Very short run
