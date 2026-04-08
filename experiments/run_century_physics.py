"""
Century run with physically-grounded parameters (post-overhaul).

Replaces the "tank mode" approach (Ah=5e6, prescribed wind, salinity relaxation)
with semi-implicit time stepping, dynamic wind stress, prognostic salinity, and
reduced artificial damping.

Usage:
    python experiments/run_century_physics.py                    # Fresh start
    python experiments/run_century_physics.py --resume year_042  # Resume from checkpoint

AMOC diagnostics are logged every 500 steps and streamfunction saved every 5 years.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp

from chronos_esm import main, io as model_io, data, log_utils
from chronos_esm.ocean import diagnostics as ocean_diagnostics


# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("outputs/century_physics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Ocean parameters: moderate relaxation from tank mode.
# Keep original hyperdiffusion/wind caps in atmosphere (proven stable).
# Structural improvements (Forward Euler + del^4, dynamic wind, prognostic salinity)
# are the key changes; ocean params can be relaxed gently.
OCEAN_PARAMS = dict(
    Ah=1.0e6,             # Ocean viscosity: 5x reduction (tank=5e6)
    shapiro_strength=0.5,  # Moderate filtering (tank=0.9)
    kappa_gm=2000.0,      # GM bolus (tank=3000)
    kappa_h=1000.0,        # Horizontal diffusivity (tank=2000)
)

PHYS_PARAMS = {"qc_ref": 0.4915, "epsilon_smooth": 4.71e-2}

CP = 1004.0  # Specific heat at constant pressure


# ============================================================================
# Diagnostics
# ============================================================================

def compute_diagnostics(state, step, steps_per_year, E0, Z0, start_time,
                        atlantic_mask, total_steps):
    """Compute and print all diagnostics."""
    elapsed = time.time() - start_time
    rate = step / max(elapsed, 1.0)
    year = step / steps_per_year

    # Atmospheric
    t_global = float(state.atmos.temp.mean())
    max_wind = float(jnp.sqrt(state.atmos.u**2 + state.atmos.v**2).max())
    max_vort = float(jnp.abs(state.atmos.vorticity).max())

    # Trade winds (10-20° latitude band, zonal mean)
    lat = jnp.linspace(-90, 90, state.atmos.u.shape[0])
    trade_mask = (jnp.abs(lat) >= 10.0) & (jnp.abs(lat) <= 20.0)
    trade_u = float(jnp.mean(state.atmos.u[trade_mask, :]))

    # Energy / Enstrophy
    cos_lat = jnp.cos(jnp.deg2rad(lat))[:, None]
    KE = 0.5 * (state.atmos.u**2 + state.atmos.v**2)
    IE = CP * state.atmos.temp
    E_now = float(jnp.sum((KE + IE) * cos_lat))
    Z_now = float(jnp.sum(state.atmos.vorticity**2 * cos_lat))
    dE_pct = 100.0 * (E_now - E0) / (abs(E0) + 1e-10)
    dZ_pct = 100.0 * (Z_now - Z0) / (abs(Z0) + 1e-10)

    # AMOC
    amoc = ocean_diagnostics.compute_amoc(state.ocean, atlantic_mask)
    amoc_upper = float(amoc["upper_cell_26N"])
    amoc_lower = float(amoc["lower_cell_26N"])

    # Ocean
    na_diag = ocean_diagnostics.compute_amoc_diagnostics(state.ocean, atlantic_mask)
    na_sst = na_diag["north_atlantic_sst"]
    mean_salt = na_diag["global_mean_salinity"]

    # Ice
    ice_extent = float(jnp.sum(state.ice.concentration > 0.15))

    print(
        f"Step {step}/{total_steps} (Yr {year:.1f}) [{rate:.1f} s/s] | "
        f"T={t_global:.1f}K Wind={max_wind:.0f}m/s Vort={max_vort:.1e} "
        f"Trade={trade_u:.1f}m/s | "
        f"dE={dE_pct:+.3f}% dZ={dZ_pct:+.3f}% | "
        f"AMOC={amoc_upper:.1f}/{amoc_lower:.1f}Sv "
        f"NA_SST={na_sst:.1f}K S={mean_salt:.2f}psu Ice={ice_extent:.0f}cells"
    )

    return t_global, dE_pct


# ============================================================================
# Main
# ============================================================================

def main_run():
    parser = argparse.ArgumentParser(description="Century run with realistic physics")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (e.g. 'year_042' or full path)")
    parser.add_argument("--years", type=float, default=100.0,
                        help="Total years to simulate")
    args = parser.parse_args()

    print("=" * 70)
    print("  CHRONOS-ESM: Century Run with Physics Overhaul")
    print("  Semi-implicit | Dynamic wind stress | Prognostic salinity")
    print("=" * 70)

    # Initialize or resume
    dt_coupled = 900.0  # DT_OCEAN
    steps_per_year = int(365 * 24 * 3600 / dt_coupled)
    start_year = 0

    if args.resume:
        # Resolve resume path
        if Path(args.resume).exists():
            resume_path = Path(args.resume)
        else:
            resume_path = OUTPUT_DIR / f"{args.resume}.nc"
            if not resume_path.exists():
                # Try century_run directory
                resume_path = Path("outputs/century_run") / f"{args.resume}.nc"

        if not resume_path.exists():
            print(f"ERROR: Resume file not found: {resume_path}")
            sys.exit(1)

        print(f"Resuming from: {resume_path}")
        state = model_io.load_state_from_netcdf(str(resume_path))

        # Extract start year from filename
        name = resume_path.stem
        if name.startswith("year_"):
            start_year = int(name.split("_")[1])
        else:
            start_year = int(state.time / (365 * 24 * 3600))
    else:
        print("Fresh initialization...")
        state = main.init_model()

    years_to_run = args.years - start_year
    if years_to_run <= 0:
        print(f"Already at year {start_year}, nothing to run.")
        return

    total_steps = int(years_to_run * steps_per_year)

    params = main.ModelParams(
        co2_ppm=280.0,
        solar_constant=1361.0,
        mask=data.load_bathymetry_mask(nz=15),
    )
    reg = main.regrid.Regridder()
    logger = log_utils.setup_logger("century_physics", OUTPUT_DIR / "model_physics.log")

    # Pre-compute Atlantic mask for AMOC diagnostics
    atlantic_mask = ocean_diagnostics.create_atlantic_mask()

    # Initial diagnostics
    lat = jnp.linspace(-90, 90, state.atmos.temp.shape[0])
    cos_lat = jnp.cos(jnp.deg2rad(lat))[:, None]
    KE0 = 0.5 * (state.atmos.u**2 + state.atmos.v**2)
    IE0 = CP * state.atmos.temp
    E0 = float(jnp.sum((KE0 + IE0) * cos_lat))
    Z0 = float(jnp.sum(state.atmos.vorticity**2 * cos_lat))
    print(f"Initial Energy: {E0:.6e} | Initial Enstrophy: {Z0:.6e}")
    print(f"Running {years_to_run:.0f} years ({total_steps} steps) from Year {start_year}...")
    print()

    start_time = time.time()

    for step in range(1, total_steps + 1):
        state = main.step_coupled(
            state, params, reg,
            physics_params=PHYS_PARAMS,
            **OCEAN_PARAMS,
        )

        # Diagnostic logging every 500 steps
        if step % 500 == 0:
            state.ocean.temp.block_until_ready()
            t_global, dE_pct = compute_diagnostics(
                state, step, steps_per_year, E0, Z0,
                start_time, atlantic_mask, total_steps,
            )

            # NaN check
            if jnp.isnan(t_global):
                logger.error(f"NaN at step {step}! Saving emergency checkpoint.")
                model_io.save_state_to_netcdf(
                    state, str(OUTPUT_DIR / "emergency_nan.nc"))
                sys.exit(1)

            # Energy drift warning
            if abs(dE_pct) > 5.0:
                logger.warning(f"Energy drift {dE_pct:.2f}% at step {step}")

        # Yearly checkpoint
        if step % steps_per_year == 0:
            year_idx = start_year + (step // steps_per_year)
            fname = OUTPUT_DIR / f"year_{year_idx:03d}.nc"
            model_io.save_state_to_netcdf(state, str(fname))
            logger.info(f"Saved checkpoint: {fname}")

            # AMOC streamfunction every 5 years
            if year_idx % 5 == 0:
                amoc = ocean_diagnostics.compute_amoc(state.ocean, atlantic_mask)
                import numpy as np
                np.save(str(OUTPUT_DIR / f"amoc_sf_year_{year_idx:03d}.npy"),
                        np.array(amoc["streamfunction"]))
                logger.info(f"Saved AMOC streamfunction for Year {year_idx}")

    # Final save
    model_io.save_state_to_netcdf(state, str(OUTPUT_DIR / "final_state.nc"))
    elapsed = time.time() - start_time
    print(f"\nCentury run complete! Total wall time: {elapsed/3600:.1f} hours")


if __name__ == "__main__":
    main_run()
