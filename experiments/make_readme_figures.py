"""
Generate the README validation dashboard (PNG maps + scorecard) from a model state.

Scores the state against WOA18 (ocean) + ERA5 (atmosphere) and writes, for every
surface field, a bias map (model / obs / model-obs) and a zonal-mean comparison
into docs/figures/, plus a scorecard table. It then rewrites the section of
README.md between the markers:

    <!-- VALIDATION:START -->  ...  <!-- VALIDATION:END -->

so the README always shows the latest simulation's validation. Re-run this after
a run (or on a checkpoint), commit, and review on GitHub.

Usage:
    python experiments/make_readme_figures.py [path/to/state.nc] [--label "year 50"]
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob as _glob  # noqa: E402

from chronos_esm import data, io  # noqa: E402
from chronos_esm.validation import grid, metrics, obs, plots  # noqa: E402
from chronos_esm.validation import scorecard as sc  # noqa: E402
from experiments.validate_control import canonical_fields, mean_fields  # noqa: E402

plots.plt.switch_backend("Agg")

OUT = "docs/figures"
README = "README.md"
START, END = "<!-- VALIDATION:START -->", "<!-- VALIDATION:END -->"

# (key, title, group) in display order.
VARS = [
    ("sst", "Sea-surface temperature", "ocean"),
    ("sss", "Sea-surface salinity", "ocean"),
    ("t2m", "2 m air temperature", "atmos"),
    ("u_sfc", "Surface zonal wind", "atmos"),
    ("v_sfc", "Surface meridional wind", "atmos"),
    ("precip", "Precipitation", "atmos"),
    ("mslp", "Mean sea-level pressure", "atmos"),
]


def update_readme_section(block):
    with open(README) as f:
        txt = f.read()
    if START in txt and END in txt:
        pre = txt.split(START)[0]
        post = txt.split(END)[1]
        txt = f"{pre}{START}\n{block}\n{END}{post}"
    else:  # append a new section if markers are missing
        txt = txt.rstrip() + f"\n\n## Validation dashboard\n{START}\n{block}\n{END}\n"
    with open(README, "w") as f:
        f.write(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "state",
        nargs="?",
        default="outputs/century_physics/final_state.nc",
        help="a state .nc file OR a glob (e.g. 'outputs/.../snap_*.nc'); "
        "a glob is averaged into a climatology",
    )
    ap.add_argument(
        "--label", default=None, help="human label for the state (e.g. 'year 50')"
    )
    ap.add_argument(
        "--no-readme",
        action="store_true",
        help="write figures only, don't touch README",
    )
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # Accept a single file or a glob. A glob is averaged into a climatology, which
    # removes weather noise and the spin-up transient from the dashboard metrics.
    paths = (
        sorted(_glob.glob(args.state))
        if any(c in args.state for c in "*?[")
        else [args.state]
    )
    if not paths:
        raise SystemExit(f"No files match {args.state!r}")
    label = args.label or (
        f"mean of {len(paths)} states"
        if len(paths) > 1
        else os.path.basename(args.state)
    )
    print(f"Scoring {len(paths)} state(s) matching {args.state} ({label}) ...")
    states = [io.load_state_from_netcdf(p) for p in paths]
    mf = mean_fields(states) if len(states) > 1 else canonical_fields(states[0])
    state = states[-1]  # last state used for the ocean mask + AMOC diagnostics

    ocean_surface = obs.woa18_surface()
    try:
        era5 = obs.era5_climatology_fields()
    except Exception as e:  # noqa: BLE001
        print(f"ERA5 unavailable ({e}); ocean-only.")
        era5 = None
    obs_spec = sc.assemble_obs(ocean_surface=ocean_surface, era5=era5)

    omask = np.asarray(data.load_bathymetry_mask(nz=15)).astype(bool)
    if omask.ndim == 3:
        omask = omask[0]

    rows = []
    for key, title, group in VARS:
        if key not in obs_spec or key not in mf:
            continue
        units, scale, cmap, _ = sc._DISPLAY[key]
        m = np.asarray(mf[key]) * scale
        nlat, nlon = m.shape
        mlat, mlon = grid.model_lat(nlat), grid.model_lon(nlon)
        o = obs_spec[key]
        obs_m = grid.regrid_to_model(o["obs"], o["lat"], o["lon"], mlat, mlon) * scale
        if group == "ocean":
            m = np.where(omask, m, np.nan)  # ocean cells only
        w = grid.area_weights(mlat)
        s = metrics.area_weighted_stats(m, obs_m, w)
        t = metrics.taylor_stats(m, obs_m, w)
        rows.append((key, units, s, t["std_ratio"]))

        plots.bias_map(
            m,
            obs_m,
            mlat,
            mlon,
            title,
            units,
            os.path.join(OUT, f"biasmap_{key}.png"),
            cmap=cmap,
        )
        plots.zonal_mean_plot(
            metrics.zonal_mean(m),
            metrics.zonal_mean(obs_m),
            mlat,
            title,
            units,
            os.path.join(OUT, f"zonal_{key}.png"),
        )

    # --- AMOC: Atlantic overturning streamfunction (map) + value vs RAPID ---
    amoc_max = None
    try:
        import jax.numpy as jnp

        from chronos_esm.config import OCEAN_DEPTH_CENTERS
        from chronos_esm.ocean import diagnostics as ocean_diag

        amoc = ocean_diag.compute_amoc(state.ocean, ocean_mask=jnp.asarray(omask))
        psi = np.asarray(amoc["streamfunction"])  # (nz, ny) Sv
        latd = grid.model_lat(psi.shape[1])
        depth = np.asarray(OCEAN_DEPTH_CENTERS)
        amoc_max = float(np.nanmax(psi))
        plots.amoc_streamfunction(
            psi,
            latd,
            depth,
            os.path.join(OUT, "amoc_streamfunction.png"),
            amoc_max=amoc_max,
            rapid=obs.AMOC_RAPID["value"],
        )
    except Exception as e:  # noqa: BLE001
        print(f"AMOC map skipped ({e})")

    # AMOC time-series from yearly checkpoints, if a multi-year run exists.
    amoc_ts = None
    yearly = sorted(_glob.glob("outputs/century_physics/year_*.nc"))
    if len(yearly) >= 2:
        try:
            import jax.numpy as jnp

            from chronos_esm.ocean import diagnostics as ocean_diag

            yrs, vals = [], []
            for p in yearly:
                stp = io.load_state_from_netcdf(p)
                d = ocean_diag.compute_amoc_diagnostics(stp.ocean)
                yrs.append(
                    int("".join(ch for ch in os.path.basename(p) if ch.isdigit()) or 0)
                )
                vals.append(float(d.get("amoc_max", np.nan)))
            plots.drift_timeseries(
                np.array(yrs),
                {"AMOC max": (np.array(vals), "Sv")},
                os.path.join(OUT, "amoc_timeseries.png"),
                title="AMOC evolution",
            )
            amoc_ts = True
        except Exception as e:  # noqa: BLE001
            print(f"AMOC time-series skipped ({e})")

    # --- scorecard table (markdown) ---
    tbl = [
        "| field | units | bias | RMSE | corr | std ratio | n |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key, units, s, sr in rows:
        tbl.append(
            f"| {key} | {units} | {s['bias']:.2f} | {s['rmse']:.2f} | "
            f"{s['corr']:.2f} | {sr:.2f} | {s['n']} |"
        )
    table_md = "\n".join(tbl)
    with open(os.path.join(OUT, "scorecard.md"), "w") as f:
        f.write(table_md + "\n")

    # --- README block (table + grouped maps) ---
    obs_src = "WOA18 + ERA5" if era5 is not None else "WOA18"
    lines = [
        # NOTE: caption uses the human `label`, not the raw `args.state` glob, so an
        # absolute scratch path (e.g. /p/tmp/$USER/...) never leaks into the public README.
        f"_Validation of the {label} against {obs_src}, "
        f"auto-generated by `experiments/make_readme_figures.py`. "
        f"A perfect model has bias 0, corr 1, std ratio 1._",
        "",
        table_md,
        "",
    ]
    for grp, header, ref in [
        ("ocean", "Ocean (vs WOA18)", ""),
        ("atmos", "Atmosphere (vs ERA5)", ""),
    ]:
        grp_keys = [k for k, _, g in VARS if g == grp and any(r[0] == k for r in rows)]
        if not grp_keys:
            continue
        lines.append(f"### {header}")
        for k in grp_keys:
            title = next(t for kk, t, _ in VARS if kk == k)
            lines.append(f"**{title}**")
            lines.append(f"![{k} bias map](docs/figures/biasmap_{k}.png)")
            lines.append(f"![{k} zonal mean](docs/figures/zonal_{k}.png)")
            lines.append("")
    if amoc_max is not None:
        lines.append("### AMOC (Atlantic overturning)")
        lines.append(
            f"Model max {amoc_max:.1f} Sv vs RAPID ~{obs.AMOC_RAPID['value']:.0f} Sv "
            "at 26.5N. NOTE this 'max' is a single-snapshot maximum of the overturning "
            "streamfunction and is **noisy**: the 26.5N cell is a clean textbook shape "
            "at any instant (smooth ~20 Sv max near 900 m, closing at the floor), but "
            "its *amplitude* oscillates strongly over the run while SST stays flat. The "
            "diagnostic thermal wind is negligible (~0.1 Sv, drag-damped); the AMOC is "
            "the density-driven **thermohaline closure**, which sets the overturning "
            "*instantaneously* from the small subpolar-subtropical density contrast and "
            "so has no temporal inertia (the basin net transport is ~0 -- already "
            "fixed). Time-mean AMOC is ~20 Sv; adding multi-year inertia to the closure "
            "is the planned fix (see experiments/diagnose_coupled_amoc.py, "
            "docs/prognostic_ocean_core.md)."
        )
        lines.append("![AMOC streamfunction](docs/figures/amoc_streamfunction.png)")
        if amoc_ts:
            lines.append("![AMOC time series](docs/figures/amoc_timeseries.png)")
        else:
            lines.append(
                "_(Time series appears once a multi-year run writes yearly checkpoints.)_"
            )
        lines.append("")
    block = "\n".join(lines)

    print("\n" + table_md)
    if not args.no_readme:
        update_readme_section(block)
        print(f"\nUpdated {README} validation dashboard and {OUT}/ figures.")
    else:
        print(f"\nWrote figures to {OUT}/ (README not modified).")


if __name__ == "__main__":
    main()
