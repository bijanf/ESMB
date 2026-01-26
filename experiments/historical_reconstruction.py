"""
Historical Reconstruction Experiment (1850-2025).

Runs the coupled model with time-varying forcing (CO2, Solar).
Outputs monthly means for comparison with observations.
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
from chronos_esm import forcing  # noqa: E402

from chronos_esm import main  # noqa: E402
from chronos_esm import io as model_io  # noqa: E402
from chronos_esm.atmos import dynamics as atmos_driver  # noqa: E402
from chronos_esm.coupler import state as coupled_state  # noqa: E402
from chronos_esm.land import driver as land_driver  # noqa: E402
from chronos_esm.ocean import veros_driver  # noqa: E402


def setup_historical_run():
    """Initialize model with real ICs (assumed ~1850 equilibrium or close)."""
    # Ideally should use spin-up from control run.
    # For now, start from WOA18 (Modern) but with 1850 Forcing.
    # This might cause a shock (cooling), but we accept this for the demo.
    
    print("Loading initial conditions (WOA18)...")
    temp_ic, salt_ic = data.load_initial_conditions(nz=15)

    ny, nx = temp_ic.shape[1], temp_ic.shape[2]
    nz = temp_ic.shape[0]

    ocean = veros_driver.OceanState(
        u=jnp.zeros((nz, ny, nx)),
        v=jnp.zeros((nz, ny, nx)),
        w=jnp.zeros((nz, ny, nx)),
        temp=temp_ic + 273.15,
        salt=salt_ic,
        psi=jnp.zeros((ny, nx)),
        rho=jnp.zeros((nz, ny, nx)),
        dic=jnp.ones((nz, ny, nx)) * 2000.0,
    )

    atmos = atmos_driver.init_atmos_state(ny, nx)
    sst_ic = temp_ic[0] + 273.15
    land = land_driver.init_land_state(ny, nx)
    state = coupled_state.init_coupled_state(ocean, atmos, land)
    fluxes = state.fluxes._replace(sst=sst_ic)
    state = state._replace(fluxes=fluxes)

    return state


def run_historical(start_year: float = 1850.0, end_year: float = 2025.0):
    """Run historical simulation."""
    state = setup_historical_run()

    from chronos_esm.config import DT_ATMOS, DT_OCEAN
    
    dt = DT_OCEAN
    seconds_per_year = 365 * 24 * 3600
    seconds_per_month = seconds_per_year / 12.0
    steps_per_month = int(seconds_per_month / dt)
    
    years_total = end_year - start_year
    n_months = int(years_total * 12)

    # Setup Logger
    from chronos_esm import log_utils as c_log

    output_dir = Path("outputs/historical_reconstruction")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = c_log.setup_logger("historical_recon", output_dir / "model.log")

    logger.info(f"Starting Historical Run {start_year}-{end_year} ({n_months} months)...")
    logger.info(f"Output directory: {output_dir}")

    # Save initial state
    model_io.save_state_to_netcdf(state, output_dir / "state_init.nc")

    t0 = time.time()
    regridder = main.regrid.Regridder()
    fm = forcing.forcing_manager

    # Accumulator Definition
    class Accumulator(NamedTuple):
        atmos_temp: jnp.ndarray
        atmos_u: jnp.ndarray
        atmos_v: jnp.ndarray
        atmos_q: jnp.ndarray
        ocean_temp: jnp.ndarray
        ocean_v: jnp.ndarray # Critical for AMOC
        ocean_w: jnp.ndarray
        sst: jnp.ndarray
        precip: jnp.ndarray

    def extract_fields(s):
        return Accumulator(
            atmos_temp=s.atmos.temp,
            atmos_u=s.atmos.u,
            atmos_v=s.atmos.v,
            atmos_q=s.atmos.q,
            ocean_temp=s.ocean.temp,
            ocean_v=s.ocean.v,
            ocean_w=s.ocean.w,
            sst=s.fluxes.sst,
            precip=s.fluxes.precip,
        )

    def zero_accumulator(s):
        fields = extract_fields(s)
        return jax.tree_util.tree_map(jnp.zeros_like, fields)

    # Step Function taking params explicitly
    def month_step_fn(carry, _):
        # Unpack
        (s, acc), current_params = carry
        
        # Step
        new_s = main.step_coupled(s, current_params, regridder)

        # Accumulate
        current_fields = extract_fields(new_s)
        new_acc = jax.tree_util.tree_map(lambda x, y: x + y, acc, current_fields)

        # Return same params to carry forward in scan
        return ((new_s, new_acc), current_params), None

    # JIT compile the monthly loop
    @jax.jit
    def run_one_month(start_state, params):
        init_acc = zero_accumulator(start_state)
        
        # Scan requires consistent carry structure
        # carry = ((state, acc), params)
        # We pass params in carry so it's available to step_fn
        
        init_carry = ((start_state, init_acc), params)
        
        (final_carry, _), _ = jax.lax.scan(
            month_step_fn, init_carry, jnp.arange(steps_per_month)
        )
        
        (final_s, final_acc) = final_carry[0]
        return final_s, final_acc

    current_state = state

    logger.info("Starting simulation loop...")

    for month_idx in range(n_months):
        current_year = start_year + month_idx / 12.0
        month_display = month_idx + 1
        
        # Get Forcing
        forcing_params = fm.get_forcing(current_year)
        
        # Convert to JAX Scalars (to avoid recompilation if seen as Python float)
        # NamedTuple expects fields matching ModelParams
        current_params = main.ModelParams(
            co2_ppm=jnp.array(forcing_params.co2_ppm),
            solar_constant=jnp.array(forcing_params.solar_constant),
            co2_increase_rate=0.0, # Handled by updating ppm directly
            mask=None
        )
        
        t_start = time.time()
        
        # Run Month
        current_state, accumulated = run_one_month(current_state, current_params)
        
        current_state.ocean.temp.block_until_ready()
        t_end = time.time()
        
        t_mean = float(jnp.mean(current_state.atmos.temp))
        logger.info(
            f"Year {current_year:.2f} (Month {month_display}) | "
            f"CO2: {float(forcing_params.co2_ppm):.1f} | "
            f"T_mean: {t_mean:.2f}K | "
            f"Time: {t_end - t_start:.2f}s"
        )

        # Save Annual Mean ? Or Monthly?
        # User wants historical reconstruction. Monthly is better for variability.
        # But 175 years * 12 months = 2100 files.
        # Let's save Monthly means, but maybe zip them or just keep them.
        
        if month_display % 12 == 0 or month_display == 1: # Save Jan and consequent annual
             # Save logic (keep it simpler for this prompt)
             pass

        # Save Yearly Mean (Accumulate externally if needed, but here we output every month)
        # Actually let's save every month for now.
        
        # Compute Means
        means = jax.tree_util.tree_map(lambda x: x / steps_per_month, accumulated)
        
        mean_filename = output_dir / f"mean_{int(current_year)}_{month_display%12:02d}.nc"
        # (Construct minimal state object for IO)
        save_state = current_state.atmos._replace(temp=means.atmos_temp) 
        # Full save is expensive. Let's just save the full state object as usual (optimized IO)
        # Reconstruct full object (expensive boilerplate omitted for brevity, using simple logic)
        
        # We need a proper State object for save_state_to_netcdf
        # Update current_state with means just for saving
        # Careful: current_state should preserve prognostic for next step (which it does, it's new_s)
        # But for saving we want means.
        
        state_to_save = current_state # Placeholder, ideally we swap in means
        # Swapping means manually like in production_control.py:
        mean_ocean = current_state.ocean._replace(
            temp=means.ocean_temp,
            v=means.ocean_v,
            w=means.ocean_w
        )
        mean_atmos = current_state.atmos._replace(
            temp=means.atmos_temp,
            u=means.atmos_u,
            v=means.atmos_v,
            q=means.atmos_q
        )
        mean_fluxes = current_state.fluxes._replace(
            sst=means.sst, precip=means.precip
        )
        state_to_save = current_state._replace(
            ocean=mean_ocean, atmos=mean_atmos, fluxes=mean_fluxes
        )
        
        model_io.save_state_to_netcdf(state_to_save, mean_filename)
        
        if t_mean > 350 or t_mean < 200:
            logger.error(f"Crash detected at {current_year}")
            break

    logger.info("Historical Run Complete.")

if __name__ == "__main__":
    run_historical()
