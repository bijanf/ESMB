"""
Chronos-ESM with JCM atmosphere using jax.lax.scan for single-compilation.

v2: Proper bulk heat/freshwater fluxes + interactive SST coupling.

The entire coupled step (atm + ocean + flux exchange) is compiled once
via jax.lax.scan, avoiding repeated JIT compilation overhead.
SST is updated at chunk boundaries (~5 days) by rebuilding the JCM step
function with new forcing (JAX caches the compilation).

Usage:
    python experiments/run_jcm_coupled.py [--years 5]
"""

import argparse
import os
import sys
import time
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import jax_datetime as jdt

import jcm
from jcm.model import Model
from jcm.forcing import ForcingData, default_forcing
from jcm.diffusion import DiffusionFilter
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics_interface import dynamics_state_to_physics_state

from chronos_esm.config import OCEAN_GRID, EARTH_RADIUS, DT_OCEAN
from chronos_esm.ocean import veros_driver as ocean_driver
from chronos_esm.ocean import diagnostics as ocean_diagnostics

OUTPUT_DIR = Path("outputs/jcm_coupled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEPS_PER_YEAR = int(365.25 * 86400 / DT_OCEAN)
CHUNK_SIZE = 500  # steps per scan chunk (~5.2 days)

# Physical constants for bulk flux computation
RHO_AIR = 1.225       # kg/m^3
CP_AIR = 1004.0       # J/kg/K
LV = 2.5e6            # J/kg latent heat of vaporization
SIGMA = 5.67e-8       # W/m^2/K^4 Stefan-Boltzmann
C_D = 1.2e-3          # bulk drag coefficient
GUSTINESS = 1.0       # m/s wind speed floor


def compute_qsat(T):
    """Saturation specific humidity [kg/kg] from temperature [K]."""
    # Bolton formula for saturation vapor pressure
    T_C = T - 273.15
    e_sat = 611.2 * jnp.exp(17.67 * T_C / (T_C + 243.5))  # Pa
    return 0.622 * e_sat / (101325.0 - 0.378 * e_sat)


def create_ocean_grid():
    """Create ocean grid spacing arrays."""
    ny, nx = OCEAN_GRID.nlat, OCEAN_GRID.nlon
    dy = jnp.pi * EARTH_RADIUS / ny
    lat = jnp.linspace(-90, 90, ny)
    cos_lat = jnp.maximum(jnp.cos(jnp.deg2rad(lat)), 0.05)
    dx = (2 * jnp.pi * EARTH_RADIUS * cos_lat[:, None]) / nx
    nz = 15
    dz = jnp.ones(nz) * (5000.0 / nz)
    return dx, dy, dz


def build_jcm_step(jcm_model, forcing):
    """Build a JCM step function with given forcing (captures forcing in closure)."""
    from dinosaur.time_integration import step_with_filters
    step_fn_factory = jcm_model._get_step_fn_factory(forcing)
    raw_step = step_fn_factory()
    return step_with_filters(raw_step, jcm_model.filters)


def build_coupled_step(jcm_model, jcm_step, dx, dy, dz, ocean_params, ocean_mask=None, atm_transmission=0.65):
    """Build the scan-compatible coupled step function."""
    ny, nx = OCEAN_GRID.nlat, OCEAN_GRID.nlon
    nz = 15
    if ocean_mask is None:
        ocean_mask = jnp.ones((ny, nx))

    lat_arr = jnp.linspace(-90, 90, ny)
    lat_rad = jnp.deg2rad(lat_arr)
    cos_lat = jnp.cos(lat_rad)[:, None]

    # Solar insolation reaching ocean surface
    # TOA: 340 W/m^2 global mean. Atmosphere absorbs ~23% (ozone, water vapor).
    # Clouds reflect ~25% on average. Net at surface: ~52% of TOA.
    sw_toa = 340.0  # W/m^2 global mean
    # atm_transmission passed as function argument
    sw_profile = sw_toa * atm_transmission * jnp.maximum(jnp.cos(lat_rad), 0.0)[:, None]
    sw_profile = jnp.broadcast_to(sw_profile, (ny, nx))

    def coupled_step(carry, step_idx):
        jcm_state, ocean_state, sim_time = carry

        # 1. Step JCM atmosphere
        new_jcm_state = jcm_step(jcm_state)

        # 2. Extract surface fields from JCM (modal -> nodal)
        physics_state = dynamics_state_to_physics_state(
            new_jcm_state, jcm_model.primitive
        )
        # JCM shape: (levels, lon, lat) -> transpose bottom level to (lat, lon)
        u_surf = physics_state.u_wind[-1].T * 0.7
        v_surf = physics_state.v_wind[-1].T * 0.7
        temp_atm = physics_state.temperature[-1].T       # K
        q_atm = physics_state.specific_humidity[-1].T     # g/kg -> kg/kg
        q_atm = jnp.maximum(q_atm * 1e-3, 0.0)           # g/kg to kg/kg

        sst = ocean_state.temp[0]  # K
        wind_mag = jnp.sqrt(u_surf**2 + v_surf**2 + GUSTINESS**2)

        # ============================================================
        # 3. BULK AERODYNAMIC HEAT FLUXES (replaces Newtonian coupling)
        # ============================================================
        # Sensible heat: H = rho * cp * Cd * |V| * (SST - T_atm)
        sensible = RHO_AIR * CP_AIR * C_D * wind_mag * (sst - temp_atm)

        # Latent heat: LE = rho * Lv * Cd * |V| * (q_sat(SST) - q_atm)
        q_sat_sst = compute_qsat(sst)
        latent = RHO_AIR * LV * C_D * wind_mag * (q_sat_sst - q_atm)
        latent = jnp.maximum(latent, 0.0)  # Only upward (evaporation)

        # Shortwave: simple profile (absorbed by ocean)
        albedo = 0.06  # ocean albedo
        sw_net = (1 - albedo) * sw_profile

        # Longwave: net = down - up
        lw_down = 0.8 * SIGMA * temp_atm**4
        lw_up = 0.97 * SIGMA * sst**4
        lw_net = lw_down - lw_up

        # Net heat into ocean (positive = warming)
        heat_flux = sw_net + lw_net - sensible - latent
        heat_flux = jnp.clip(heat_flux, -800.0, 800.0)

        # ============================================================
        # 4. FRESHWATER FLUX
        # ============================================================
        # Evaporation: E = LE / Lv [kg/m^2/s]
        evaporation = latent / LV

        # Precipitation: simple estimate from column humidity convergence
        # P ≈ q_atm * wind_mag * scale (very rough)
        # Better: use moisture relaxation P = (q - q_crit) / tau_precip
        q_crit = 0.8 * q_sat_sst
        tau_precip = 4.0 * 3600.0  # 4-hour condensation timescale
        precip = jnp.maximum(q_atm - q_crit, 0.0) / tau_precip * 1000.0  # rough kg/m^2/s

        # Net freshwater = precip - evap (positive = freshening ocean)
        fw_flux = precip - evaporation
        fw_flux = jnp.clip(fw_flux, -0.01, 0.01)

        # ============================================================
        # 5. WIND STRESS (with prescribed blend)
        # ============================================================
        years_elapsed = sim_time / (365.25 * 86400)
        day_of_year = (sim_time % (365.25 * 86400)) / 86400.0

        tau_x_dyn = jnp.clip(RHO_AIR * C_D * wind_mag * u_surf, -0.3, 0.3)
        tau_y_dyn = jnp.clip(RHO_AIR * C_D * wind_mag * v_surf, -0.3, 0.3)

        season_shift = 10.0 * jnp.sin(2.0 * jnp.pi * (day_of_year - 80.0) / 365.0)
        lat_eff = lat_rad - jnp.deg2rad(season_shift)
        tau_x_pre = jnp.broadcast_to((-0.08 * jnp.sin(6.0 * lat_eff))[:, None], (ny, nx))
        tau_y_pre = jnp.zeros((ny, nx))

        # Keep prescribed wind as floor — JCM aquaplanet winds (~1 m/s)
        # are too weak to drive AMOC. Use whichever is stronger at each point.
        tau_x = jnp.where(jnp.abs(tau_x_dyn) > jnp.abs(tau_x_pre), tau_x_dyn, tau_x_pre)
        tau_y = jnp.where(jnp.abs(tau_y_dyn) > jnp.abs(tau_y_pre), tau_y_dyn, tau_y_pre)

        # ============================================================
        # 6. STEP OCEAN (apply mask to fluxes — no forcing on land)
        # ============================================================
        heat_flux = heat_flux * ocean_mask
        fw_flux = fw_flux * ocean_mask
        tau_x = tau_x * ocean_mask
        tau_y = tau_y * ocean_mask
        carbon_flux = jnp.zeros((ny, nx))
        new_ocean = ocean_driver.step_ocean(
            ocean_state,
            surface_fluxes=(heat_flux, fw_flux, carbon_flux),
            wind_stress=(tau_x, tau_y),
            dx=dx, dy=dy, dz=dz,
            nz=nz, ny=ny, nx=nx,
            mask=ocean_mask, dt=DT_OCEAN,
            **ocean_params,
        )

        new_sim_time = sim_time + DT_OCEAN
        new_carry = (new_jcm_state, new_ocean, new_sim_time)

        # AMOC: integrate Atlantic meridional velocity
        # Simple proxy: max of cumulative v*dx*dz in Atlantic (280-360E, 34S-80N)
        lon_idx = jnp.arange(nx)
        atlantic_lon = ((lon_idx * 360.0 / nx) >= 280) | ((lon_idx * 360.0 / nx) <= 20)
        atlantic_lat = (lat_arr >= -34) & (lat_arr <= 80)
        atl_mask = atlantic_lat[:, None] & atlantic_lon[None, :]
        v_atl = jnp.where(atl_mask[None, :, :], new_ocean.v, 0.0)
        # Zonally integrate v, then cumsum in depth (Sv)
        dx_1d = 2 * jnp.pi * EARTH_RADIUS * jnp.cos(lat_rad) / nx
        v_transport = jnp.sum(v_atl * dx_1d[None, :, None], axis=2)  # (nz, ny)
        dz_val = 5000.0 / nz
        sf = -jnp.cumsum(v_transport * dz_val, axis=0) / 1e6  # Sv
        lat_26N = jnp.argmin(jnp.abs(lat_arr - 26.5))
        # Skip surface Ekman layer (top 3 levels = 1000m) to measure
        # deep thermohaline AMOC, not wind-driven surface overturning
        amoc_26n = jnp.max(sf[3:, lat_26N])

        diagnostics = dict(
            temp_atm=jnp.mean(temp_atm),
            max_wind=jnp.sqrt(u_surf**2 + v_surf**2).max(),
            sst_mean=jnp.mean(sst),
            salt_mean=jnp.mean(new_ocean.salt[0]),
            tau_x_mean=jnp.mean(tau_x),
            heat_flux_mean=jnp.mean(heat_flux),
            fw_flux_mean=jnp.mean(fw_flux),
            sensible_mean=jnp.mean(sensible),
            latent_mean=jnp.mean(latent),
            amoc_26n=amoc_26n,
        )

        return new_carry, diagnostics

    return coupled_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", type=float, default=5.0)
    parser.add_argument("--atm-trans", type=float, default=0.65)
    parser.add_argument("--r-drag", type=float, default=0.10)
    parser.add_argument("--kappa-gm", type=float, default=2000.0)
    args = parser.parse_args()

    print("=" * 70)
    print("  CHRONOS-ESM with JCM Atmosphere + Real Earth Geography")
    print("  Spherical harmonics | IMEX RK3 | Continental boundaries")
    print("=" * 70, flush=True)

    ny, nx = OCEAN_GRID.nlat, OCEAN_GRID.nlon
    nz = 15

    # Load real terrain and forcing from JCM's built-in data
    import jcm as jcm_pkg
    jcm_data = Path(os.path.dirname(jcm_pkg.__file__)) / 'data' / 'bc' / 't30' / 'clim'

    print("Loading real Earth terrain + forcing...", flush=True)
    coords = get_speedy_coords(spectral_truncation=31)

    from jcm.terrain import TerrainData
    terrain = TerrainData.from_file(str(jcm_data / 'terrain.nc'), coords=coords)

    forcing = ForcingData.from_file(str(jcm_data / 'forcing.nc'), coords=coords)

    # Derive ocean mask from JCM terrain (fmask: 1=land, 0=ocean in JCM)
    # Our convention: 1=ocean, 0=land; transpose from (lon,lat) to (lat,lon)
    ocean_mask = (1.0 - terrain.fmask).T
    ocean_mask = jnp.where(ocean_mask > 0.5, 1.0, 0.0)  # Binarize
    n_ocean = int(ocean_mask.sum())
    n_land = ocean_mask.size - n_ocean
    print(f"  Land: {n_land} cells ({100*n_land/ocean_mask.size:.0f}%), "
          f"Ocean: {n_ocean} cells ({100*n_ocean/ocean_mask.size:.0f}%)", flush=True)

    # Build JCM model with real terrain
    print("Building JCM model with real topography...", flush=True)
    jcm_model = Model(
        coords=coords, time_step=15.0,
        terrain=terrain,
        start_date=jdt.to_datetime("2000-01-01"),
    )
    jcm_state = jcm_model._prepare_initial_modal_state()

    # Initialize ocean with realistic T/S from WOA18
    print("Initializing ocean with WOA18 climatology...", flush=True)
    ocean_state = ocean_driver.init_ocean_state(nz, ny, nx)
    try:
        from chronos_esm.data import load_initial_conditions
        temp_ic, salt_ic = load_initial_conditions(nz=nz)
        # WOA18 temp is in Celsius — convert to Kelvin
        temp_ic = jnp.array(temp_ic) + 273.15
        salt_ic = jnp.array(salt_ic)
        # Fill land cells with safe defaults (avoid NaN propagation)
        mask_3d = jnp.broadcast_to(ocean_mask[None, :, :], (nz, ny, nx))
        temp_ic = jnp.where(mask_3d > 0.5, temp_ic, 280.0)  # 280K on land
        salt_ic = jnp.where(mask_3d > 0.5, salt_ic, 35.0)    # 35 psu on land
        ocean_state = ocean_state._replace(temp=temp_ic, salt=salt_ic)
        print(f"  WOA18: SST range [{float(temp_ic[0].min()):.1f}, {float(temp_ic[0].max()):.1f}] K, "
              f"S range [{float(salt_ic[0].min()):.1f}, {float(salt_ic[0].max()):.1f}] psu", flush=True)
    except Exception as e:
        print(f"  Warning: Could not load WOA18 data ({e}), using default init", flush=True)

    # Ocean grid
    dx, dy, dz = create_ocean_grid()

    ocean_params = dict(
        r_drag=args.r_drag, kappa_gm=args.kappa_gm, kappa_h=1000.0,
        kappa_bi=0.0, Ah=1.0e6, Ab=0.0,
        shapiro_strength=0.5, smag_constant=0.1,
    )
    print(f"  Params: atm_trans={args.atm_trans}, r_drag={args.r_drag}, kappa_gm={args.kappa_gm}", flush=True)

    total_steps = int(args.years * STEPS_PER_YEAR)
    n_chunks = total_steps // CHUNK_SIZE

    print(f"Running {args.years} years = {total_steps} steps "
          f"({n_chunks} chunks of {CHUNK_SIZE})", flush=True)

    # First chunk: build step fn, compile, run
    print("Building JCM step function + JIT compiling...", flush=True)
    t_compile = time.time()
    jcm_step = build_jcm_step(jcm_model, forcing)
    coupled_step = build_coupled_step(jcm_model, jcm_step, dx, dy, dz, ocean_params,
                                      ocean_mask=ocean_mask, atm_transmission=args.atm_trans)

    @jax.jit
    def run_chunk(carry):
        return jax.lax.scan(coupled_step, carry, jnp.arange(CHUNK_SIZE))

    carry = (jcm_state, ocean_state, 0.0)
    carry, diag = run_chunk(carry)
    print(f"Compiled + first chunk in {time.time()-t_compile:.1f}s", flush=True)
    print(f"  T_atm={float(diag['temp_atm'][-1]):.1f}K "
          f"SST={float(diag['sst_mean'][-1]):.1f}K "
          f"Wind={float(diag['max_wind'][-1]):.1f}m/s "
          f"H={float(diag['heat_flux_mean'][-1]):.0f}W/m2 "
          f"S={float(diag['salt_mean'][-1]):.2f}psu", flush=True)

    # Remaining chunks with SST coupling at boundaries
    t_start = time.time()
    sst_update_interval = 20  # Update SST every 20 chunks (~100 days)

    for chunk_idx in range(1, n_chunks):
        # Update SST coupling periodically
        if chunk_idx % sst_update_interval == 0:
            _, ocean_st, _ = carry
            new_sst = ocean_st.temp[0].T  # (lat,lon) -> (lon,lat)
            forcing = forcing.copy(sea_surface_temperature=new_sst)
            jcm_step = build_jcm_step(jcm_model, forcing)
            coupled_step = build_coupled_step(jcm_model, jcm_step, dx, dy, dz, ocean_params,
                                                ocean_mask=ocean_mask, atm_transmission=args.atm_trans)

            @jax.jit
            def run_chunk(carry):
                return jax.lax.scan(coupled_step, carry, jnp.arange(CHUNK_SIZE))

        carry, diag = run_chunk(carry)

        # Print diagnostics
        jcm_st, ocn_st, sim_time = carry
        years = float(sim_time) / (365.25 * 86400)
        rate = (chunk_idx + 1) * CHUNK_SIZE / max(time.time() - t_start, 1)

        print(
            f"Step {(chunk_idx+1)*CHUNK_SIZE}/{total_steps} (Yr {years:.1f}) "
            f"[{rate:.0f} s/s] | "
            f"T_atm={float(diag['temp_atm'][-1]):.1f}K "
            f"SST={float(diag['sst_mean'][-1]):.1f}K "
            f"Wind={float(diag['max_wind'][-1]):.1f}m/s "
            f"H={float(diag['heat_flux_mean'][-1]):.0f}W/m2 "
            f"FW={float(diag['fw_flux_mean'][-1]):.2e} "
            f"S={float(diag['salt_mean'][-1]):.2f}psu "
            f"tau={float(diag['tau_x_mean'][-1]):.4f}Pa "
            f"AMOC={float(diag['amoc_26n'][-1]):.1f}Sv",
            flush=True,
        )

        # Yearly checkpoint
        step_num = (chunk_idx + 1) * CHUNK_SIZE
        if step_num % STEPS_PER_YEAR < CHUNK_SIZE:
            year_num = int(years)
            ckpt_path = OUTPUT_DIR / f"year_{year_num:03d}.nc"
            import netCDF4 as nc
            with nc.Dataset(str(ckpt_path), "w") as ds:
                nz_o, ny_o, nx_o = np.array(ocn_st.temp).shape
                ds.createDimension("z", nz_o)
                ds.createDimension("y", ny_o)
                ds.createDimension("x", nx_o)
                for name in ["temp", "salt", "u", "v"]:
                    v = ds.createVariable(f"ocean_{name}", "f4", ("z", "y", "x"))
                    v[:] = np.array(getattr(ocn_st, name))
                ds.sim_time = float(sim_time)
            print(f"  Saved: {ckpt_path}", flush=True)

    elapsed = time.time() - t_start
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f}min)", flush=True)


if __name__ == "__main__":
    main()
