"""
Orchestrates obs-vs-model scoring into a printable scorecard (+ optional figures).

Canonical model field keys (all 2-D on the model grid, model units):
    sst    [degC]        sea-surface temperature   -> WOA18 surface T
    sss    [psu]         sea-surface salinity       -> WOA18 surface S
    t2m    [K]           near-surface air temp      -> ERA5 2m temperature
    u_sfc  [m/s]         near-surface zonal wind    -> ERA5 10m u
    v_sfc  [m/s]         near-surface merid. wind   -> ERA5 10m v
    precip [kg/m^2/s]    precipitation rate         -> ERA5 total precipitation
    mslp   [Pa]          mean sea-level pressure    -> ERA5 MSL pressure

Each variable is reported in a human-friendly display unit (e.g. precip in
mm/day, mslp in hPa). Correlation is scale-invariant; bias/RMSE are reported in
the display unit.
"""

import os

import numpy as np

from chronos_esm.validation import grid, metrics, plots

# Per-variable display config: (display_units, scale model->display, cmap, kind)
_DISPLAY = {
    "sst":    ("degC",   1.0,            "RdBu_r", "ocean"),
    "sss":    ("psu",    1.0,            "RdBu_r", "ocean"),
    "t2m":    ("K",      1.0,            "RdBu_r", "atmos"),
    "u_sfc":  ("m/s",    1.0,            "RdBu_r", "atmos"),
    "v_sfc":  ("m/s",    1.0,            "RdBu_r", "atmos"),
    "precip": ("mm/day", 86400.0,        "BrBG",   "atmos"),
    "mslp":   ("hPa",    0.01,           "RdBu_r", "atmos"),
}


def assemble_obs(ocean_surface=None, era5=None):
    """Build {var: dict(obs, lat, lon)} from raw obs bundles (native grids)."""
    spec = {}
    if ocean_surface is not None:
        spec["sst"] = dict(obs=ocean_surface["sst"], lat=ocean_surface["lat"],
                           lon=ocean_surface["lon"])
        spec["sss"] = dict(obs=ocean_surface["sss"], lat=ocean_surface["lat"],
                           lon=ocean_surface["lon"])
    if era5 is not None:
        for key, src in (("t2m", "t2m"), ("u_sfc", "u10"), ("v_sfc", "v10"),
                         ("precip", "precip"), ("mslp", "msl")):
            spec[key] = dict(obs=era5[src], lat=era5["lat"], lon=era5["lon"])
    return spec


def run_scorecard(model_fields, obs_spec, outdir=None, make_figures=False):
    """Score each model field against its regridded obs.

    model_fields : {var: 2-D array on the model grid (model units)}
    obs_spec     : {var: dict(obs=native 2-D, lat=, lon=)}
    Returns a list of row dicts (one per scored variable) and writes figures
    into `outdir` when make_figures is True.
    """
    rows = []
    figures = []
    taylor_entries = []

    if make_figures and outdir:
        os.makedirs(outdir, exist_ok=True)

    for var, mfield in model_fields.items():
        if var not in obs_spec:
            continue
        mfield = np.asarray(mfield, float)
        nlat, nlon = mfield.shape
        mlat, mlon = grid.model_lat(nlat), grid.model_lon(nlon)
        w = grid.area_weights(mlat)

        units, scale, cmap, kind = _DISPLAY.get(var, (None, 1.0, "RdBu_r", "atmos"))

        o = obs_spec[var]
        obs_m = grid.regrid_to_model(o["obs"], o["lat"], o["lon"], mlat, mlon)

        m_disp = mfield * scale
        o_disp = obs_m * scale

        s = metrics.area_weighted_stats(m_disp, o_disp, w)
        t = metrics.taylor_stats(m_disp, o_disp, w)
        rows.append(dict(var=var, units=units, **s,
                         std_ratio=t["std_ratio"], crmse_ratio=t["crmse_ratio"]))
        taylor_entries.append(dict(name=var, corr=s["corr"], std_ratio=t["std_ratio"]))

        if make_figures and outdir:
            figures.append(plots.bias_map(
                m_disp, o_disp, mlat, mlon, var.upper(), units,
                os.path.join(outdir, f"biasmap_{var}.pdf"), cmap=cmap))
            figures.append(plots.zonal_mean_plot(
                metrics.zonal_mean(m_disp), metrics.zonal_mean(o_disp), mlat,
                var.upper(), units, os.path.join(outdir, f"zonal_{var}.pdf")))

    if make_figures and outdir and taylor_entries:
        figures.append(plots.taylor_diagram(
            taylor_entries, os.path.join(outdir, "taylor.pdf")))

    return rows, figures


def format_scorecard(rows):
    """Render the scorecard rows as a fixed-width text table."""
    if not rows:
        return "(no variables scored)"
    hdr = (f"{'var':7s} {'units':7s} {'bias':>9s} {'rmse':>9s} "
           f"{'corr':>6s} {'std_ratio':>9s} {'mean_mod':>9s} {'mean_obs':>9s} {'n':>6s}")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['var']:7s} {r['units']:7s} {r['bias']:9.3f} {r['rmse']:9.3f} "
            f"{r['corr']:6.2f} {r['std_ratio']:9.2f} {r['mean_model']:9.3f} "
            f"{r['mean_obs']:9.3f} {r['n']:6d}")
    return "\n".join(lines)
