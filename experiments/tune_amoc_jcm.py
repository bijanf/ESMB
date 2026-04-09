"""
Gradient descent optimization of AMOC in JCM-coupled Chronos-ESM.

Uses TBPTT (Truncated Backpropagation Through Time) to optimize
5 parameters that control AMOC strength:
  1. atm_transmission  — SST gradient driver (solar reaching ocean)
  2. wind_amp          — prescribed wind stress amplitude
  3. r_drag            — ocean bottom friction
  4. kappa_gm          — Gent-McWilliams eddy diffusivity
  5. kappa_h           — horizontal diffusivity

Target: AMOC at 26.5N = 17 Sv (observed value).

Usage:
    python experiments/tune_amoc_jcm.py [--horizon 100] [--epochs 100]
"""

import argparse
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import jax_datetime as jdt

import jcm
from jcm.model import Model
from jcm.forcing import ForcingData, default_forcing
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics_interface import dynamics_state_to_physics_state
from dinosaur.time_integration import step_with_filters

from chronos_esm.config import OCEAN_GRID, EARTH_RADIUS, DT_OCEAN
from chronos_esm.ocean import veros_driver as ocean_driver
from chronos_esm.ocean import diagnostics
from chronos_esm.ocean.utils import soft_clip

OUTPUT_DIR = Path("outputs/tune_amoc")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Physical constants
RHO_AIR = 1.225
CP_AIR = 1004.0
LV = 2.5e6
SIGMA = 5.67e-8
C_D = 1.2e-3
GUSTINESS = 1.0

# Targets
TARGET_AMOC = 17.0  # Sv at 26.5N
TARGET_SST_GRADIENT = 25.0  # K equator-pole
TARGET_SALT = 35.0  # psu
TARGET_SST_MEAN = 288.0  # K

STEPS_PER_YEAR = int(365.25 * 86400 / DT_OCEAN)


def compute_qsat(T):
    T_C = T - 273.15
    e_sat = 611.2 * jnp.exp(17.67 * T_C / (T_C + 243.5))
    return 0.622 * e_sat / (101325.0 - 0.378 * e_sat)


def create_ocean_grid():
    ny, nx = OCEAN_GRID.nlat, OCEAN_GRID.nlon
    dy = jnp.pi * EARTH_RADIUS / ny
    lat = jnp.linspace(-90, 90, ny)
    cos_lat = jnp.maximum(jnp.cos(jnp.deg2rad(lat)), 0.05)
    dx = (2 * jnp.pi * EARTH_RADIUS * cos_lat[:, None]) / nx
    dz = jnp.ones(15) * (5000.0 / 15)
    return dx, dy, dz


def constrain_params(raw_params):
    """Map unconstrained raw params to physical ranges."""
    raw_atm, raw_wind, raw_r, raw_kgm, raw_kh = raw_params
    return dict(
        atm_transmission=jax.nn.sigmoid(raw_atm),     # (0, 1), ~0.70
        wind_amp=-jax.nn.softplus(raw_wind),           # < 0, ~-0.08
        r_drag=jax.nn.softplus(raw_r) + 1e-4,         # > 0, ~0.05
        kappa_gm=jax.nn.softplus(raw_kgm) + 1.0,      # > 1, ~2000
        kappa_h=jax.nn.softplus(raw_kh) + 10.0,        # > 10, ~1000
    )


def init_raw_params():
    """Initialize raw params for real geography run.
    atm_transmission=0.90 (warm enough for real geography)
    r_drag=5.0 (high friction to moderate AMOC)
    kappa_gm=5000 (strong eddy mixing)
    """
    raw_atm = jnp.log(0.90 / (1.0 - 0.90))             # sigmoid^-1(0.90)
    raw_wind = jnp.log(jnp.exp(0.08) - 1.0)              # softplus^-1(0.08)
    raw_r = jnp.log(jnp.exp(5.0 - 1e-4) - 1.0)          # softplus^-1(5.0)
    raw_kgm = 5000.0 - 1.0                                # softplus ≈ identity for large x
    raw_kh = 1000.0 - 10.0
    return (jnp.array(raw_atm), jnp.array(raw_wind),
            jnp.array(raw_r), jnp.array(raw_kgm), jnp.array(raw_kh))


def build_model():
    """Build JCM atmosphere + ocean model with real Earth geography."""
    import os
    import jcm as jcm_pkg
    from jcm.terrain import TerrainData

    ny, nx = OCEAN_GRID.nlat, OCEAN_GRID.nlon
    nz = 15

    # Load real terrain and forcing
    jcm_data = Path(os.path.dirname(jcm_pkg.__file__)) / 'data' / 'bc' / 't30' / 'clim'
    coords = get_speedy_coords(spectral_truncation=31)
    terrain = TerrainData.from_file(str(jcm_data / 'terrain.nc'), coords=coords)
    forcing = ForcingData.from_file(str(jcm_data / 'forcing.nc'), coords=coords)

    # Ocean mask from JCM terrain (1=ocean, 0=land)
    ocean_mask = jnp.where((1.0 - terrain.fmask).T > 0.5, 1.0, 0.0)

    # Build JCM model with real topography
    jcm_model = Model(coords=coords, time_step=15.0, terrain=terrain,
                      start_date=jdt.to_datetime("2000-01-01"))
    jcm_state = jcm_model._prepare_initial_modal_state()

    # Build JCM step function with real forcing
    step_fn_factory = jcm_model._get_step_fn_factory(forcing)
    raw_step = step_fn_factory()
    jcm_step = step_with_filters(raw_step, jcm_model.filters)

    ocean_state = ocean_driver.init_ocean_state(nz, ny, nx)
    dx, dy, dz = create_ocean_grid()

    return jcm_model, jcm_step, jcm_state, ocean_state, dx, dy, dz, ocean_mask, forcing


def build_loss_fn(jcm_model, jcm_step, dx, dy, dz, horizon, ocean_mask=None):
    """Build the differentiable loss function for AMOC optimization."""
    ny, nx = OCEAN_GRID.nlat, OCEAN_GRID.nlon
    nz = 15
    if ocean_mask is None:
        ocean_mask = jnp.ones((ny, nx))

    lat_arr = jnp.linspace(-90, 90, ny)
    lat_rad = jnp.deg2rad(lat_arr)
    sw_toa = 340.0

    # Masks for SST gradient computation
    equator_mask = (jnp.abs(lat_arr) < 15.0).astype(jnp.float32)
    polar_mask = (jnp.abs(lat_arr) > 60.0).astype(jnp.float32)

    def loss_fn(raw_params, carry_init):
        params = constrain_params(raw_params)

        # Unpack
        atm_trans = params["atm_transmission"]
        wind_amp = params["wind_amp"]
        r_drag = params["r_drag"]
        kappa_gm = params["kappa_gm"]
        kappa_h = params["kappa_h"]

        # Solar profile with tunable transmission
        sw_profile = sw_toa * atm_trans * jnp.maximum(jnp.cos(lat_rad), 0.0)[:, None]
        sw_profile = jnp.broadcast_to(sw_profile, (ny, nx))

        def coupled_step(carry, step_idx):
            jcm_state, ocean_state, sim_time = carry

            # Step JCM atmosphere
            new_jcm_state = jcm_step(jcm_state)

            # Extract surface fields
            physics_state = dynamics_state_to_physics_state(
                new_jcm_state, jcm_model.primitive)
            u_surf = physics_state.u_wind[-1].T * 0.7
            v_surf = physics_state.v_wind[-1].T * 0.7
            temp_atm = physics_state.temperature[-1].T
            q_atm = jnp.maximum(physics_state.specific_humidity[-1].T * 1e-3, 0.0)

            sst = ocean_state.temp[0]
            wind_mag = jnp.sqrt(u_surf**2 + v_surf**2 + GUSTINESS**2)

            # Bulk heat fluxes
            sensible = RHO_AIR * CP_AIR * C_D * wind_mag * (sst - temp_atm)
            q_sat_sst = compute_qsat(sst)
            latent = jnp.maximum(RHO_AIR * LV * C_D * wind_mag * (q_sat_sst - q_atm), 0.0)
            sw_net = (1 - 0.06) * sw_profile
            lw_down = 0.8 * SIGMA * temp_atm**4
            lw_up = 0.97 * SIGMA * sst**4
            heat_flux = soft_clip(sw_net + lw_down - lw_up - sensible - latent, -800.0, 800.0)

            # Freshwater
            q_crit = 0.8 * q_sat_sst
            precip = jnp.maximum(q_atm - q_crit, 0.0) / (4.0 * 3600.0) * 1000.0
            evaporation = latent / LV
            fw_flux = soft_clip(precip - evaporation, -0.01, 0.01)

            # Wind stress with tunable amplitude
            years_elapsed = sim_time / (365.25 * 86400)
            day_of_year = (sim_time % (365.25 * 86400)) / 86400.0
            tau_x_dyn = soft_clip(RHO_AIR * C_D * wind_mag * u_surf, -0.3, 0.3)
            tau_y_dyn = soft_clip(RHO_AIR * C_D * wind_mag * v_surf, -0.3, 0.3)

            season_shift = 10.0 * jnp.sin(2.0 * jnp.pi * (day_of_year - 80.0) / 365.0)
            lat_eff = lat_rad - jnp.deg2rad(season_shift)
            tau_x_pre = jnp.broadcast_to((wind_amp * jnp.sin(6.0 * lat_eff))[:, None], (ny, nx))
            tau_y_pre = jnp.zeros((ny, nx))

            alpha = jnp.clip(years_elapsed / 5.0, 0.0, 1.0)
            tau_x = alpha * tau_x_dyn + (1 - alpha) * tau_x_pre
            tau_y = alpha * tau_y_dyn + (1 - alpha) * tau_y_pre

            # Mask fluxes to ocean cells only
            heat_flux = heat_flux * ocean_mask
            fw_flux = fw_flux * ocean_mask
            tau_x = tau_x * ocean_mask
            tau_y = tau_y * ocean_mask

            # Step ocean with tunable params
            new_ocean = ocean_driver.step_ocean(
                ocean_state,
                surface_fluxes=(heat_flux, fw_flux, jnp.zeros((ny, nx))),
                wind_stress=(tau_x, tau_y),
                dx=dx, dy=dy, dz=dz,
                nz=nz, ny=ny, nx=nx, mask=ocean_mask, dt=DT_OCEAN,
                r_drag=r_drag, kappa_gm=kappa_gm, kappa_h=kappa_h,
                kappa_bi=0.0, Ah=1.0e6, Ab=0.0,
                shapiro_strength=0.5, smag_constant=0.1,
            )

            return (new_jcm_state, new_ocean, sim_time + DT_OCEAN), None

        @jax.checkpoint
        def checkpointed_step(carry, step_idx):
            return coupled_step(carry, step_idx)

        final_carry, _ = jax.lax.scan(checkpointed_step, carry_init,
                                       jnp.arange(horizon))

        _, ocean_final, _ = final_carry

        # === LOSS COMPUTATION (gradient-chain-aware) ===

        # 1. AMOC: compute streamfunction, skip surface Ekman (top 3 levels)
        #    Use soft-max (logsumexp) instead of jnp.max for smooth gradients
        amoc_result = diagnostics.compute_amoc(ocean_final, dz=5000.0/nz)
        sf = amoc_result["streamfunction"]  # (nz, ny)
        lat_26N_idx = jnp.argmin(jnp.abs(lat_arr - 26.5))
        sf_deep = sf[3:, lat_26N_idx]  # Below 1000m (skip Ekman)
        # Soft-max: differentiable approximation to max
        softmax_temp = 1.0  # temperature parameter
        amoc_26n = softmax_temp * jax.nn.logsumexp(sf_deep / softmax_temp)
        loss_amoc = (amoc_26n - TARGET_AMOC) ** 2

        # 2. Ocean kinetic energy (smooth, always-differentiable proxy for circulation)
        ocean_ke = jnp.mean(ocean_final.u**2 + ocean_final.v**2)
        target_ke = 1e-4  # Rough target for realistic circulation
        loss_ke = (ocean_ke - target_ke) ** 2 * 1e8

        # 3. SST gradient loss
        sst = ocean_final.temp[0]
        sst_eq = jnp.sum(sst * equator_mask[:, None]) / (jnp.sum(equator_mask) * nx + 1e-10)
        sst_pole = jnp.sum(sst * polar_mask[:, None]) / (jnp.sum(polar_mask) * nx + 1e-10)
        sst_gradient = sst_eq - sst_pole
        loss_sst_grad = (sst_gradient - TARGET_SST_GRADIENT) ** 2

        # 4. Global mean SST loss (most direct for atm_transmission)
        sst_mean = jnp.mean(sst)
        loss_sst_mean = (sst_mean - TARGET_SST_MEAN) ** 2

        # 5. Salinity loss
        salt_mean = jnp.mean(ocean_final.salt)
        loss_salt = (salt_mean - TARGET_SALT) ** 2

        # 6. Regularization
        loss_reg = 0.01 * (
            (params["r_drag"] - 5.0) ** 2
            + ((params["kappa_gm"] - 5000) / 1000) ** 2
            + ((params["kappa_h"] - 1000) / 500) ** 2
        )

        # Weight SST_mean heavily — it's the most direct & smooth target
        total = (1.0 * loss_amoc + 0.5 * loss_ke + 0.1 * loss_sst_grad +
                 2.0 * loss_sst_mean + 0.5 * loss_salt + loss_reg)

        aux = dict(
            total_loss=total, amoc_26n=amoc_26n, sst_gradient=sst_gradient,
            salt_mean=salt_mean, sst_mean=sst_mean, ocean_ke=ocean_ke,
            atm_trans=params["atm_transmission"], wind_amp=params["wind_amp"],
            r_drag=params["r_drag"], kappa_gm=params["kappa_gm"],
            kappa_h=params["kappa_h"], final_carry=final_carry,
        )
        return total, aux

    return loss_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=100, help="TBPTT horizon (steps)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--updates-per-epoch", type=int, default=50, help="Updates per epoch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--spinup-years", type=float, default=1.0, help="Spinup years")
    args = parser.parse_args()

    print("=" * 70)
    print("  AMOC Gradient Descent Optimization (JCM-coupled)")
    print(f"  Target: AMOC={TARGET_AMOC} Sv | Horizon={args.horizon} steps")
    print(f"  Epochs={args.epochs} | Updates/epoch={args.updates_per_epoch}")
    print("=" * 70, flush=True)

    # Build model with real geography
    print("Building model with real Earth geography...", flush=True)
    jcm_model, jcm_step, jcm_state, ocean_state, dx, dy, dz, ocean_mask, forcing = build_model()
    print(f"  Ocean cells: {int(ocean_mask.sum())}/{ocean_mask.size}", flush=True)

    # Load warm ocean state from checkpoint (with real geography)
    warm_ckpt = Path("outputs/jcm_coupled/year_018.nc")
    if warm_ckpt.exists():
        import netCDF4 as nc
        print(f"Loading warm ocean state from {warm_ckpt}...", flush=True)
        ds = nc.Dataset(str(warm_ckpt))
        ocean_state = ocean_driver.OceanState(
            u=jnp.array(ds["ocean_u"][:]),
            v=jnp.array(ds["ocean_v"][:]),
            w=jnp.zeros_like(jnp.array(ds["ocean_u"][:])),
            temp=jnp.array(ds["ocean_temp"][:]),
            salt=jnp.array(ds["ocean_salt"][:]),
            psi=jnp.zeros((OCEAN_GRID.nlat, OCEAN_GRID.nlon)),
            rho=jnp.zeros_like(jnp.array(ds["ocean_temp"][:])),
            dic=jnp.zeros_like(jnp.array(ds["ocean_temp"][:])),
        )
        sim_time = float(ds.sim_time) if hasattr(ds, "sim_time") else 48.0 * 365.25 * 86400
        ds.close()
        print(f"  SST={float(ocean_state.temp[0].mean()):.1f}K "
              f"S={float(ocean_state.salt[0].mean()):.2f}psu", flush=True)

        # Equilibrate JCM atmosphere with ocean state (1 year)
        # Use real forcing (already loaded) — JCM sees real SST climatology
        print("Equilibrating JCM + ocean (1 year)...", flush=True)
        from experiments.run_jcm_coupled import build_jcm_step, build_coupled_step
        jcm_step = build_jcm_step(jcm_model, forcing)  # Use real forcing
        default_ocean_params = dict(
            r_drag=0.10, kappa_gm=2000.0, kappa_h=1000.0,
            kappa_bi=0.0, Ah=1.0e6, Ab=0.0,
            shapiro_strength=0.5, smag_constant=0.1,
        )
        eq_coupled = build_coupled_step(jcm_model, jcm_step, dx, dy, dz,
                                        default_ocean_params, ocean_mask=ocean_mask)

        @jax.jit
        def equilibrate(carry):
            return jax.lax.scan(eq_coupled, carry, jnp.arange(500))

        carry = (jcm_state, ocean_state, sim_time)
        eq_chunks = 70  # 70 x 500 = 35000 steps ≈ 1 year
        sst_update_every = 20  # Update JCM SST forcing every 20 chunks
        for i in range(eq_chunks):
            carry, diag = equilibrate(carry)

            # Update JCM SST forcing from ocean (same as production run)
            if (i + 1) % sst_update_every == 0:
                _, oc, _ = carry
                new_sst = oc.temp[0].T
                forcing = forcing.copy(sea_surface_temperature=new_sst)
                jcm_step = build_jcm_step(jcm_model, forcing)
                eq_coupled = build_coupled_step(jcm_model, jcm_step, dx, dy, dz,
                                                default_ocean_params, ocean_mask=ocean_mask)

                @jax.jit
                def equilibrate(carry):
                    return jax.lax.scan(eq_coupled, carry, jnp.arange(500))

                print(f"  Equil chunk {i+1}/{eq_chunks}: "
                      f"SST={float(oc.temp[0].mean()):.1f}K "
                      f"S={float(oc.salt[0].mean()):.2f}psu (SST updated)", flush=True)

        jcm_state, ocean_state, sim_time = carry
        print(f"  Equilibration done: SST={float(ocean_state.temp[0].mean()):.1f}K", flush=True)
    else:
        print("No warm checkpoint found — using cold start.", flush=True)
        sim_time = 0.0

    # Initialize parameters and optimizer
    raw_params = init_raw_params()
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(args.lr),
    )
    opt_state = optimizer.init(raw_params)

    # Build loss function with ocean mask
    print(f"Building loss function (horizon={args.horizon})...", flush=True)
    loss_fn = build_loss_fn(jcm_model, jcm_step, dx, dy, dz, args.horizon, ocean_mask=ocean_mask)

    # JIT compile the update step
    @jax.jit
    def update_step(raw_params, opt_state, carry):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(raw_params, carry)
        updates, new_opt_state = optimizer.update(grads, opt_state, raw_params)
        new_params = optax.apply_updates(raw_params, updates)

        # Safety: replace NaN gradients PER-PARAMETER (not all-or-nothing)
        # This ensures one bad gradient doesn't freeze all 5 parameters
        new_params = jax.tree.map(
            lambda p, old, g: jnp.where(jnp.isfinite(g), p, old),
            new_params, raw_params, grads)
        new_opt_state = jax.tree.map(
            lambda s, old: jnp.where(jnp.isfinite(loss), s, old),
            new_opt_state, opt_state)

        return new_params, new_opt_state, loss, aux

    # First compilation
    print("JIT compiling optimization step (may take a few minutes)...", flush=True)
    t0 = time.time()
    carry = (jcm_state, ocean_state, sim_time)
    raw_params, opt_state, loss, aux = update_step(raw_params, opt_state, carry)
    carry = aux["final_carry"]
    print(f"Compiled in {time.time()-t0:.1f}s. Initial loss={float(loss):.4f} "
          f"AMOC={float(aux['amoc_26n']):.2f}Sv", flush=True)

    # Training loop
    print(f"\n{'Ep':>4} {'Loss':>8} {'AMOC':>6} {'dSST':>6} {'Salt':>6} "
          f"{'SST':>6} {'atm_t':>6} {'wind':>7} {'r_dr':>7} {'kgm':>6} {'kh':>6}",
          flush=True)
    print("-" * 95, flush=True)

    t_start = time.time()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for _ in range(args.updates_per_epoch):
            raw_params, opt_state, loss, aux = update_step(raw_params, opt_state, carry)
            carry = aux["final_carry"]
            epoch_loss += float(loss)

        epoch_loss /= args.updates_per_epoch
        rate = (epoch + 1) * args.updates_per_epoch * args.horizon / (time.time() - t_start)

        print(f"{epoch:4d} {epoch_loss:8.2f} {float(aux['amoc_26n']):6.2f} "
              f"{float(aux['sst_gradient']):6.1f} {float(aux['salt_mean']):6.2f} "
              f"{float(aux['sst_mean']):6.1f} {float(aux['atm_trans']):6.3f} "
              f"{float(aux['wind_amp']):7.4f} {float(aux['r_drag']):7.5f} "
              f"{float(aux['kappa_gm']):6.0f} {float(aux['kappa_h']):6.0f} "
              f"[{rate:.0f}s/s]", flush=True)

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            params = constrain_params(raw_params)
            ckpt = {k: float(v) for k, v in params.items()}
            ckpt["epoch"] = epoch + 1
            ckpt["loss"] = epoch_loss
            ckpt["amoc"] = float(aux["amoc_26n"])
            np.save(OUTPUT_DIR / f"params_epoch_{epoch+1:04d}.npy", ckpt)
            print(f"  Saved checkpoint epoch {epoch+1}", flush=True)

    # Final results
    params = constrain_params(raw_params)
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print(f"  AMOC: {float(aux['amoc_26n']):.2f} Sv (target: {TARGET_AMOC})")
    print(f"  SST gradient: {float(aux['sst_gradient']):.1f} K (target: {TARGET_SST_GRADIENT})")
    for k, v in params.items():
        print(f"  {k}: {float(v):.6f}")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
