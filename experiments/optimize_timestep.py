"""
Optimize Timestep Experiment.
Runs short simulations with varying DT_ATMOS to find the largest stable timestep.
"""

# import importlib # Unused
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp


# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import config, data, main  # noqa: E402
from chronos_esm.atmos import dynamics as atmos_driver  # noqa: E402
from chronos_esm.coupler import state as coupled_state  # noqa: E402
from chronos_esm.ocean import veros_driver  # noqa: E402


def setup_model():
    """Initialize model with real ICs."""
    print("Loading initial conditions (WOA18)...")
    temp_ic, salt_ic = data.load_initial_conditions(nz=15)

    # Create Ocean State
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


def run_test(dt_val, steps):
    """Run simulation with specific DT."""
    print(f"\nTesting DT_ATMOS = {dt_val}s for {steps} steps...")

    # Monkeypatch
    config.DT_ATMOS = dt_val
    main.DT_ATMOS = dt_val
    atmos_driver.DT_ATMOS = dt_val

    # Re-initialize to be safe (though state doesn't depend on DT usually)
    state = setup_model()

    regridder = main.regrid.Regridder()
    mask = data.load_bathymetry_mask()
    params = main.ModelParams(mask=mask)

    # Define step function closure to capture new DT
    # We need to reload main or redefine step_coupled?
    # main.step_coupled uses DT_ATMOS from global scope of main.
    # Since we patched main.DT_ATMOS, it should work IF we re-JIT.

    # Force re-JIT by defining a new function
    @jax.jit
    def step_fn(carry, _):
        s = carry
        new_s = main.step_coupled(s, params, regridder)
        return new_s, None

    t0 = time.time()

    try:
        # Run
        final_state, _ = jax.lax.scan(step_fn, state, jnp.arange(steps))
        final_state.ocean.temp.block_until_ready()

        t_end = time.time()
        duration = t_end - t0

        # Check Stability
        t_mean = float(jnp.mean(final_state.atmos.temp))
        u_max = float(jnp.max(jnp.abs(final_state.atmos.u)))

        is_stable = True
        if jnp.isnan(t_mean) or t_mean > 350.0 or t_mean < 150.0:
            is_stable = False
            print(f"  -> Unstable! T_mean={t_mean}")
        elif u_max > 200.0:
            is_stable = False
            print(f"  -> Unstable! Max Wind={u_max}")
        else:
            print(f"  -> Stable. T_mean={t_mean:.2f} K, Max Wind={u_max:.2f} m/s")

        # Performance
        simulated_time = steps * dt_val
        speedup = simulated_time / duration
        print(f"  -> Wall time: {duration:.2f}s. Speed: {speedup:.2f}x real-time")

        return is_stable, speedup

    except Exception as e:
        print(f"  -> Crashed: {e}")
        return False, 0.0


if __name__ == "__main__":
    # Test values
    # Current is 30s. Try increasing.
    dt_values = [30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0]

    results = {}

    for dt in dt_values:
        # Run for 1 day of simulation
        steps = int(24 * 3600 / dt)
        is_stable, speed = run_test(dt, steps)
        results[dt] = (is_stable, speed)

        if not is_stable:
            print("Stopping search due to instability.")
            break

    print("\nSummary:")
    for dt, (stable, speed) in results.items():
        status = "PASS" if stable else "FAIL"
        print(f"DT={dt:6.1f}s: {status} (Speed: {speed:.1f}x)")
