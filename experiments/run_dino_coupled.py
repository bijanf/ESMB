"""Coupled control run: dinosaur multi-level atmosphere <-> Chronos ocean.

Sequential (lagged) coupling at a fixed interval:
  ocean SST  --regrid lin->Gauss-->  DinoAtmosphere.step(SST)  -->  surface winds/T
  surface fields --regrid Gauss->lin--> bulk fluxes (wind stress, SW/LW/sensible/
  latent, net heat) --> step_ocean for the interval.

The atmosphere is the SST-coupled dinosaur dycore (chronos_esm/atmos/dino_atmos),
which performs genuine baroclinic instability and carries prognostic moisture.

This script is the CONTROL-RUN harness: it checkpoints (ocean state + dinosaur modal
state) every `--ckpt-every-days`, resumes cleanly from any checkpoint, and writes a
TIME-MEAN of the atmosphere's surface fields (u_sfc/v_sfc/t2m/precip/mslp) into the
saved state so the existing validation dashboard
(experiments/make_readme_figures.py / validate_control.py) scores the *dinosaur*
atmosphere rather than the unused single-level fields.

    # fresh 100-year control run, checkpoint yearly:
    python experiments/run_dino_coupled.py --years 100
    # resume from year 42 (day 15330) and continue:
    python experiments/run_dino_coupled.py --years 100 --resume 15330
    # quick smoke test (20 days, checkpoint every 7):
    python experiments/run_dino_coupled.py --days 20 --ckpt-every-days 7

    # score a finished run against WOA18 + ERA5:
    python experiments/make_readme_figures.py "outputs/dino_control/state_d*.nc" \
        --label "dino control"
"""
import argparse
import os
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm import main, io  # noqa: E402
from chronos_esm.ocean import veros_driver  # noqa: E402
from chronos_esm.atmos import physics as aphys  # noqa: E402
from chronos_esm.atmos.dino_atmos import (  # noqa: E402
    DinoAtmosphere, save_state as dino_save, load_state as dino_load)
from chronos_esm.config import (OCEAN_GRID, OCEAN_DZ, EARTH_RADIUS, DT_OCEAN,  # noqa: E402
                                ALBEDO_OCEAN)

RHO_AIR, CD = 1.2, 1.3e-3
LAT_LIN = np.linspace(-90, 90, OCEAN_GRID.nlat)
DAYS_PER_YEAR = 365


def make_regridders(lat_gauss):
    """1-D latitude interpolation between the linear (ocean/atmos) grid and the
    dinosaur Gaussian grid; longitudes coincide (both 96 pts at 0,3.75,...)."""
    def lin_to_gauss(f_lin):                # (nlat_lin, nlon) -> (nlon, nlat_gauss)
        f = np.asarray(f_lin)
        out = np.stack([np.interp(lat_gauss, LAT_LIN, f[:, j]) for j in range(f.shape[1])], axis=0)
        return out                          # (nlon, nlat_gauss)

    def gauss_to_lin(f_g):                  # (nlon, nlat_gauss) -> (nlat_lin, nlon)
        f = np.asarray(f_g)
        out = np.stack([np.interp(LAT_LIN, lat_gauss, f[j, :]) for j in range(f.shape[0])], axis=1)
        return out                          # (nlat_lin, nlon)
    return lin_to_gauss, gauss_to_lin


_WLAT = np.cos(np.deg2rad(LAT_LIN))[:, None]


def ocean_fluxes(sst_K, u_sfc, v_sfc, t_air_K, q_air, precip_atm, balance_heat=True,
                 ocean_mask=None):
    """Bulk surface fluxes on the linear grid, consistent with the atmosphere's
    own near-surface humidity and precipitation. Returns (net_heat W/m2,
    fw kg/m2/s, tau_x Pa, tau_y Pa)."""
    # annual-mean-ish shortwave into the ocean (insolation * transmission * (1-albedo))
    insol = np.asarray(aphys.compute_solar_insolation(jnp.asarray(LAT_LIN) * np.pi / 180.0,
                                                      day_of_year=80.0))[:, None]
    sw_net = np.maximum(insol, 0.0) * (1.0 - ALBEDO_OCEAN)
    lw_down = 0.8 * 5.67e-8 * np.maximum(t_air_K - 10.0, 150.0) ** 4
    lw_up = 0.98 * 5.67e-8 * sst_K ** 4
    # turbulent fluxes using the ATMOSPHERE's actual near-surface humidity q_air
    # (so latent heat / evaporation are consistent with the moisture budget).
    sens, lat = aphys.compute_surface_fluxes(jnp.asarray(t_air_K), jnp.asarray(q_air),
                                             jnp.asarray(u_sfc), jnp.asarray(v_sfc),
                                             jnp.asarray(sst_K))
    sens, lat = np.asarray(sens), np.asarray(lat)
    net_heat = sw_net + lw_down - lw_up - sens - lat
    if balance_heat:
        # Heat-flux adjustment: remove the area-weighted mean so the control run has
        # no net OCEAN heating/cooling (prevents global SST drift while preserving
        # the spatial flux pattern). Balance over OCEAN cells ONLY: the heat is
        # applied only to ocean cells (via step_ocean's mask), so including land
        # cells -- which carry an unphysical net_heat from the land "SST" values --
        # in the mean leaves a residual net ocean cooling and a spurious cold SST
        # drift. Falling back to the all-cell mean only if no mask is supplied.
        w = np.broadcast_to(_WLAT, net_heat.shape)
        if ocean_mask is not None:
            m = np.asarray(ocean_mask).astype(bool)
            ocean_mean = np.sum(net_heat[m] * w[m]) / np.sum(w[m])
        else:
            ocean_mean = np.sum(net_heat * w) / np.sum(w)
        net_heat = net_heat - ocean_mean
    net_heat = np.clip(net_heat, -1500.0, 1500.0)
    evap = lat / 2.5e6
    fw = precip_atm - evap  # real P - E
    if balance_heat and ocean_mask is not None:
        # Freshwater-flux adjustment (same ocean-only balance as the heat flux):
        # remove the net ocean-mean P-E so there is no net surface freshwater
        # source. veros_driver already renormalizes the 3D VOLUME-mean salinity,
        # but a net surface P-E imbalance still redistributes salt and drifts the
        # SURFACE salinity; balancing P-E over ocean cells holds SSS steady too.
        wf = np.broadcast_to(_WLAT, fw.shape)
        mf = np.asarray(ocean_mask).astype(bool)
        fw = fw - np.sum(fw[mf] * wf[mf]) / np.sum(wf[mf])
    wind_mag = np.maximum(np.sqrt(u_sfc ** 2 + v_sfc ** 2), 1.0)
    tau_x = np.clip(RHO_AIR * CD * wind_mag * u_sfc, -0.3, 0.3)
    tau_y = np.clip(RHO_AIR * CD * wind_mag * v_sfc, -0.3, 0.3)
    return net_heat, fw, tau_x, tau_y


def _ckpt_paths(outdir, day):
    """(netcdf, dino-npz) checkpoint paths for an absolute simulation day."""
    base = os.path.join(outdir, f"state_d{day:06d}")
    return base + ".nc", base + "_dino.npz"


def _resume_day(resume_arg, outdir):
    """Resolve a --resume value (an integer day, or a .nc path) to an absolute day."""
    if resume_arg is None:
        return None
    if os.path.exists(resume_arg) and resume_arg.endswith(".nc"):
        m = re.search(r"state_d(\d+)\.nc$", os.path.basename(resume_arg))
        if not m:
            raise ValueError(f"cannot parse day from {resume_arg}")
        return int(m.group(1))
    return int(resume_arg)  # treat as a day number


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=5.0, help="total simulated years")
    ap.add_argument("--days", type=int, default=0,
                    help="total simulated days (overrides --years if > 0; for smoke tests)")
    ap.add_argument("--interval", type=int, default=1, help="coupling interval [days]")
    ap.add_argument("--ckpt-every-days", type=int, default=DAYS_PER_YEAR,
                    help="checkpoint cadence [days] (default 365 = yearly)")
    ap.add_argument("--outdir", default="outputs/dino_control")
    ap.add_argument("--resume", default=None,
                    help="resume from an absolute day number or a state_d*.nc path")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    interval = args.interval
    # absolute TOTAL length of the run (a resumed job continues *toward* this, as in
    # run_century_physics: `--years 200 --resume <day>` runs to year 200, not +200).
    end_day = args.days if args.days > 0 else int(round(args.years * DAYS_PER_YEAR))

    # --- build the model + dinosaur atmosphere -----------------------------------
    state = main.init_model(ocean_ic="woa")
    ocean_mask_3d, surface_mask = main.ocean_masks(nz=state.ocean.u.shape[0])
    atm = DinoAtmosphere()
    lin_to_gauss, gauss_to_lin = make_regridders(atm.lat_deg)

    start_day = _resume_day(args.resume, args.outdir)
    if start_day is not None:
        nc, npz = _ckpt_paths(args.outdir, start_day)
        if not (os.path.exists(nc) and os.path.exists(npz)):
            raise FileNotFoundError(f"resume checkpoint missing: {nc} / {npz}")
        resumed = io.load_state_from_netcdf(nc)
        state = state._replace(ocean=resumed.ocean)   # real ocean from the checkpoint
        dino_state = dino_load(npz)                    # dinosaur modal state
        print(f"Resumed from day {start_day} ({start_day / DAYS_PER_YEAR:.2f} yr): {nc}")
    else:
        start_day = 0
        sst0_g = lin_to_gauss(np.asarray(state.ocean.temp[0]))   # WOA SST on the dino grid
        dino_state = atm.initial_state(sst0_g)                    # near-equilibrium init

    if start_day >= end_day:
        print(f"start day {start_day} >= target {end_day}; nothing to do.")
        return

    # --- ocean grid metrics (as in main.py) --------------------------------------
    dy_ocn = (np.pi * EARTH_RADIUS) / OCEAN_GRID.nlat
    cos_lat_ocn = np.maximum(np.cos(np.deg2rad(LAT_LIN)), 0.05)
    dx_ocn = jnp.asarray((2 * np.pi * EARTH_RADIUS * cos_lat_ocn[:, None]) / OCEAN_GRID.nlon)
    dz_ocn = jnp.asarray(OCEAN_DZ)
    subs = int(round(86400.0 * interval / DT_OCEAN))  # ocean substeps per interval

    @jax.jit
    def ocean_interval(ocean, fluxes, wind):
        def body(oc, _):
            return veros_driver.step_ocean(
                oc, surface_fluxes=fluxes, wind_stress=wind, dx=dx_ocn, dy=dy_ocn,
                dz=dz_ocn, nz=ocean.u.shape[0], mask=surface_mask,
                ocean_mask_3d=ocean_mask_3d), None
        oc, _ = jax.lax.scan(body, ocean, None, length=subs)
        return oc

    omask = np.asarray(surface_mask).astype(bool)
    wlat = np.cos(np.deg2rad(LAT_LIN))[:, None]

    def gmean(f):
        f = np.asarray(f); w = np.broadcast_to(wlat, f.shape)
        ok = np.isfinite(f)  # area-weight only over valid (e.g. ocean) cells
        return float(np.sum(f[ok] * w[ok]) / np.sum(w[ok]))

    # --- running TIME-MEAN accumulator of surface fields (for scoring) -----------
    acc_keys = ("u_sfc", "v_sfc", "t2m", "q", "precip", "mslp")

    def new_acc():
        return {k: np.zeros((OCEAN_GRID.nlat, OCEAN_GRID.nlon)) for k in acc_keys}, 0

    acc, nacc = new_acc()

    def save_checkpoint(day):
        """Persist ocean+modal state and inject the windowed time-mean atmosphere
        surface fields so the validation dashboard scores the dinosaur atmosphere."""
        nc, npz = _ckpt_paths(args.outdir, day)
        n = max(nacc, 1)
        mslp = np.maximum(acc["mslp"] / n, 1.0)
        atmos_score = state.atmos._replace(
            temp=jnp.asarray(acc["t2m"] / n),
            u=jnp.asarray(acc["u_sfc"] / n),
            v=jnp.asarray(acc["v_sfc"] / n),
            q=jnp.asarray(acc["q"] / n),
            ln_ps=jnp.log(jnp.asarray(mslp)),            # mslp = exp(ln_ps) with phi_s=0
            phi_s=jnp.zeros_like(jnp.asarray(mslp)),
        )
        fluxes_score = state.fluxes._replace(precip=jnp.asarray(acc["precip"] / n))
        score_state = state._replace(atmos=atmos_score, fluxes=fluxes_score,
                                     time=float(day * 86400))
        io.save_state_to_netcdf(score_state, nc)
        dino_save(dino_state, npz)
        print(f"  [checkpoint] day {day} ({day / DAYS_PER_YEAR:.2f} yr) "
              f"-> {nc} (+{npz}); time-mean of {n} samples", flush=True)

    print(f"Coupled dinosaur<->ocean control run: days {start_day}->{end_day} "
          f"(interval {interval}d, {subs} ocean substeps/interval, "
          f"checkpoint every {args.ckpt_every_days}d)", flush=True)

    day = start_day
    n_intervals = (end_day - start_day) // interval
    for it in range(1, n_intervals + 1):
        sst_lin = np.asarray(state.ocean.temp[0])                       # K
        sst_g = lin_to_gauss(sst_lin)
        dino_state = atm.step(dino_state, sst_g, n_days=interval)
        diag = atm.diagnostics(dino_state)
        u_sfc = gauss_to_lin(diag["u_sfc"]); v_sfc = gauss_to_lin(diag["v_sfc"])
        t_air = gauss_to_lin(diag["t_sfc"]); q_air = gauss_to_lin(diag["q_sfc"])
        precip_a = gauss_to_lin(diag["precip"]); mslp_a = gauss_to_lin(diag["mslp"])
        nh, fw, tx, ty = ocean_fluxes(sst_lin, u_sfc, v_sfc, t_air, q_air, precip_a,
                                      ocean_mask=omask)
        fluxes = (jnp.asarray(nh), jnp.asarray(fw), jnp.zeros_like(jnp.asarray(nh)))
        state = state._replace(ocean=ocean_interval(state.ocean,
                                                    fluxes, (jnp.asarray(tx), jnp.asarray(ty))))
        day += interval

        # accumulate the windowed time-mean (linear grid) for scoring
        for k, val in (("u_sfc", u_sfc), ("v_sfc", v_sfc), ("t2m", t_air),
                       ("q", q_air), ("precip", precip_a), ("mslp", mslp_a)):
            acc[k] += np.asarray(val)
        nacc += 1

        finite = bool(np.isfinite(diag["u"]).all()
                      and np.isfinite(np.asarray(state.ocean.temp)).all())
        if (day % 30 == 0) or (it == n_intervals) or (not finite):
            sst_oc = np.where(omask, sst_lin, np.nan)
            uup = diag["u"][atm.layers // 4].mean(axis=0); lat = atm.lat_deg
            jet = float(uup[(np.abs(lat) > 30) & (np.abs(lat) < 60)].mean())
            sss_oc = np.where(omask, np.asarray(state.ocean.salt[0]), np.nan)
            print(f"  day {day:6d} ({day / DAYS_PER_YEAR:6.2f} yr): "
                  f"SST {gmean(sst_oc) - 273.15:5.2f}C  SSS {gmean(sss_oc):5.2f}  "
                  f"|u|max {np.abs(diag['u']).max():5.1f}  midlat-jet {jet:+5.1f}  "
                  f"|curr|max {float(np.abs(np.asarray(state.ocean.u)).max()):.3f}  "
                  f"finite {finite}", flush=True)
        if not finite:
            nc, _ = _ckpt_paths(args.outdir, day)
            io.save_state_to_netcdf(state, nc.replace(".nc", "_NAN.nc"))
            raise FloatingPointError(f"non-finite state at day {day}; emergency dump written")

        if (day % args.ckpt_every_days == 0) or (it == n_intervals):
            save_checkpoint(day)
            acc, nacc = new_acc()

    print(f"done: ran to day {day} ({day / DAYS_PER_YEAR:.2f} yr). "
          f"Score with:\n  python experiments/make_readme_figures.py "
          f"'{args.outdir}/state_d*.nc' --label 'dino control'")


if __name__ == "__main__":
    main_cli()
