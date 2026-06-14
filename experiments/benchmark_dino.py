"""Benchmark the multi-level dinosaur atmosphere vs ERA5 (Phase 4).

Forces the SST-coupled dinosaur atmosphere with the (fixed) WOA SST, spins it up
to a quasi-equilibrium, averages the last AVG days into a climatology, regrids the
surface fields to the model validation grid (48x96 linear), and scores them
against ERA5 (+ WOA18 for SST/SSS) with the existing scorecard. This isolates the
atmosphere's skill given a realistic SST, for direct comparison with the
single-level baseline.

    python experiments/benchmark_dino.py --spinup 90 --avg 30
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm import main  # noqa: E402
from chronos_esm.atmos.dino_atmos import DinoAtmosphere  # noqa: E402
from chronos_esm.validation import obs, grid, metrics, scorecard as sc  # noqa: E402

LAT_LIN = np.linspace(-90, 90, 48)


def gauss_to_lin(f_g, lat_gauss):                 # (nlon,nlat_g) -> (nlat_lin,nlon)
    f = np.asarray(f_g)
    return np.stack([np.interp(LAT_LIN, lat_gauss, f[j, :]) for j in range(f.shape[0])], axis=1)


def lin_to_gauss(f_lin, lat_gauss):               # (nlat_lin,nlon) -> (nlon,nlat_g)
    f = np.asarray(f_lin)
    return np.stack([np.interp(lat_gauss, LAT_LIN, f[:, j]) for j in range(f.shape[1])], axis=0)


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spinup", type=int, default=150)
    ap.add_argument("--avg", type=int, default=90, help="climatology window (last N days)")
    ap.add_argument("--sample-every", type=int, default=2,
                    help="sample interval [days] in the averaging window; dense sampling "
                         "smooths the vigorous transient baroclinic eddies so the (weak) "
                         "time-mean surface-wind tripole is not buried in eddy noise")
    args = ap.parse_args()

    atm = DinoAtmosphere()
    lat_g = atm.lat_deg

    # WOA SST (model ocean init) on the dinosaur grid; ocean T/S for the ocean score.
    st = main.init_model(ocean_ic="woa")
    sst_lin = np.asarray(st.ocean.temp[0])          # K, (48,96)
    sss_lin = np.asarray(st.ocean.salt[0])
    sst_g = lin_to_gauss(sst_lin, lat_g)

    print(f"Spinning up dinosaur atmosphere {args.spinup} d "
          f"(avg last {args.avg} d, sampling every {args.sample_every} d) on WOA SST ...")
    state = atm.initial_state(sst_g)   # near-equilibrium init tailored to the WOA SST
    # spin-up phase (no sampling), in 5-day chunks
    for _ in range((args.spinup - args.avg) // 5):
        state = atm.step(state, sst_g, n_days=5)
    # averaging phase: sample densely so transient baroclinic eddies average out
    acc, ndone = None, 0
    for _ in range(args.avg // args.sample_every):
        state = atm.step(state, sst_g, n_days=args.sample_every)
        di = atm.diagnostics(state)
        f = {"u_sfc": di["u_sfc"], "v_sfc": di["v_sfc"], "t2m": di["t_sfc"],
             "precip": di["precip"], "mslp": di["mslp"]}
        acc = f if acc is None else {k: acc[k] + f[k] for k in acc}
        ndone += 1
    clim = {k: v / ndone for k, v in acc.items()}   # dino-grid climatology
    print(f"  averaged {ndone} samples")

    # model fields on the validation (linear 48x96) grid, model units
    mf = {
        "sst": sst_lin - 273.15,                    # degC
        "sss": sss_lin,                             # psu
        "t2m": gauss_to_lin(clim["t2m"], lat_g),    # K
        "u_sfc": gauss_to_lin(clim["u_sfc"], lat_g),
        "v_sfc": gauss_to_lin(clim["v_sfc"], lat_g),
        "precip": gauss_to_lin(clim["precip"], lat_g),   # kg/m^2/s
        "mslp": gauss_to_lin(clim["mslp"], lat_g),       # Pa (reduced to sea level)
    }

    ocean_surface = obs.woa18_surface()
    try:
        era5 = obs.era5_climatology_fields()
    except Exception as e:  # noqa: BLE001
        print(f"ERA5 unavailable ({e}); ocean-only.")
        era5 = None
    obs_spec = sc.assemble_obs(ocean_surface=ocean_surface, era5=era5)

    omask = np.asarray(main.ocean_masks(nz=st.ocean.u.shape[0])[1]).astype(bool)
    print("\n| field | units | bias | RMSE | corr | std ratio | n |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for key in ["sst", "sss", "t2m", "u_sfc", "v_sfc", "precip", "mslp"]:
        if key not in obs_spec or key not in mf:
            continue
        units, scale, _, group = sc._DISPLAY[key]
        m = np.asarray(mf[key]) * scale
        nlat, nlon = m.shape
        mlat, mlon = grid.model_lat(nlat), grid.model_lon(nlon)
        o = obs_spec[key]
        obs_m = grid.regrid_to_model(o["obs"], o["lat"], o["lon"], mlat, mlon) * scale
        if group == "ocean":
            m = np.where(omask, m, np.nan)
        w = grid.area_weights(mlat)
        s = metrics.area_weighted_stats(m, obs_m, w)
        t = metrics.taylor_stats(m, obs_m, w)
        print(f"| {key} | {units} | {s['bias']:.2f} | {s['rmse']:.2f} | "
              f"{s['corr']:.2f} | {t['std_ratio']:.2f} | {s['n']} |")


if __name__ == "__main__":
    main_cli()
