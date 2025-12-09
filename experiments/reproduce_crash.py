
"""
Reproduction script for simulation crash.
Runs a short segment of the control run to check for stability/crashes.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
# numpy as np  # Unused

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import data, main  # noqa: E402
from chronos_esm.atmos import dynamics as atmos_driver  # noqa: E402
# from chronos_esm.config import DT_ATMOS # Unused
from chronos_esm.coupler import state as coupled_state  # noqa: E402
from chronos_esm.ocean import veros_driver  # noqa: E402


def setup_control_run():
    """Initialize model with real ICs."""
    print("Loading initial conditions (WOA18)...")
    temp_ic, salt_ic = data.load_initial_conditions(nz=15)

    # Create Ocean State with loaded ICs
    ny, nx = temp_ic.shape[1], temp_ic.shape[2]
    nz = temp_ic.shape[0]

    ocean = veros_driver.OceanState(
        u=jnp.zeros((nz, ny, nx)),
        v=jnp.zeros((nz, ny, nx)),
        w=jnp.zeros((nz, ny, nx)),
        temp=temp_ic + 273.15,  # C -> K
        salt=salt_ic,
        psi=jnp.zeros((ny, nx)),
        rho=jnp.zeros((nz, ny, nx)),
        dic=jnp.ones((nz, ny, nx)) * 2000.0,
    )

    # Initialize Atmos
    atmos = atmos_driver.init_atmos_state(ny, nx)

    sst_ic = temp_ic[0] + 273.15

    # Initialize Land
    from chronos_esm.land import driver as land_driver

    land = land_driver.init_land_state(ny, nx)

    # Coupled State
    state = coupled_state.init_coupled_state(ocean, atmos, land)

    # Update fluxes SST to match
    fluxes = state.fluxes._replace(sst=sst_ic)
    state = state._replace(fluxes=fluxes)

    return state


def run_reproduction(steps: int = 100):
    """Run a short reproduction simulation."""
    print(f"Setting up reproduction run for {steps} steps...")
    state = setup_control_run()

    # Setup Logger
    from chronos_esm import log_utils as c_log

    output_dir = Path("outputs/reproduce_crash")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = c_log.setup_logger("reproduce_crash", output_dir / "model.log")

    logger.info(f"Starting Reproduction Run for {steps} steps...")

    regridder = main.regrid.Regridder()

    logger.info("Loading Land Mask...")
    mask = data.load_bathymetry_mask()
    params = main.ModelParams(mask=mask)

    # Compile the step function
    @jax.jit
    def step_fn(carry, _):
        s = carry
        new_s = main.step_coupled(s, params, regridder)
        return new_s, None

    logger.info("Starting simulation loop (JIT compiling first)...")
    t0 = time.time()

    # Run in chunks to see progress
    chunk_size = 10000
    n_chunks = steps // chunk_size

    current_state = state

    for i in range(n_chunks):
        logger.info(f"Running chunk {i+1}/{n_chunks}...")
        t_start = time.time()

        # Scan over chunk
        current_state, _ = jax.lax.scan(step_fn, current_state, jnp.arange(chunk_size))

        # Block
        current_state.ocean.temp.block_until_ready()
        t_end = time.time()

        logger.info(f"Chunk {i+1} complete in {t_end - t_start:.2f}s")

        # Check for NaNs
        if jnp.isnan(jnp.mean(current_state.atmos.temp)):
            logger.error("NaN detected in Atmos Temp!")
            break

    t1 = time.time()
    logger.info(f"Run complete in {t1-t0:.2f}s")


if __name__ == "__main__":
    # Run for 1 month (30 days * 24h * 3600s / 120s = 21600 steps)
    run_reproduction(steps=21600)
