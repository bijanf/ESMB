"""
Validation figures, styled for Nature-family submission (vector PDF).

Follows the Springer Nature artwork guide: sans-serif <=7 pt, RGB, editable
vector PDF (pdf.fonttype 42), 1-column = 3.46 in / 2-column = 7.09 in.
All functions save a vector PDF and return the path.
"""

import matplotlib

matplotlib.use("pdf")  # vector backend, no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL1, COL2 = 3.46, 7.09  # inches


def bias_map(model, obs, lat, lon, title, units, out_path,
             cmap="RdBu_r", vlim=None):
    """Three-panel model / obs / (model-obs) bias map. Returns out_path."""
    model = np.asarray(model, float)
    obs = np.asarray(obs, float)
    diff = model - obs

    if vlim is None:
        finite = diff[np.isfinite(diff)]
        vlim = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
        vlim = vlim or 1.0
    # Scale Model and Obs panels to the OBSERVED robust range (2-98th pct), shared
    # so they are directly comparable AND the obs is never flattened by a wildly
    # out-of-range model (a too-variable model simply saturates the colour scale,
    # which visually flags the problem instead of hiding it).
    ofin = obs[np.isfinite(obs)]
    if ofin.size:
        fvmin, fvmax = (float(np.nanpercentile(ofin, 2)), float(np.nanpercentile(ofin, 98)))
    else:
        fvmin, fvmax = 0.0, 1.0
    if fvmin == fvmax:
        fvmax = fvmin + 1.0

    fig, axes = plt.subplots(1, 3, figsize=(COL2, COL2 / 3.1))
    for ax, fld, ttl, vv in (
        (axes[0], model, "Model", None),
        (axes[1], obs, "Obs", None),
        (axes[2], diff, "Model - Obs", vlim),
    ):
        if vv is None:
            im = ax.pcolormesh(lon, lat, fld, cmap="viridis",
                               vmin=fvmin, vmax=fvmax, shading="auto")
        else:
            im = ax.pcolormesh(lon, lat, fld, cmap=cmap,
                               vmin=-vv, vmax=vv, shading="auto")
        ax.set_title(ttl)
        ax.set_xlabel("Longitude")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.9)
        cb.ax.tick_params(labelsize=5)
        cb.set_label(units, fontsize=5)
    axes[0].set_ylabel("Latitude")
    fig.suptitle(f"{title} [{units}]", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def zonal_mean_plot(model_zm, obs_zm, lat, title, units, out_path):
    """Model vs obs zonal-mean profile (value vs latitude). Returns out_path."""
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.8))
    ax.plot(lat, model_zm, label="Model", color="#c1272d", lw=1.0)
    ax.plot(lat, obs_zm, label="Obs", color="#0000a7", lw=1.0, ls="--")
    ax.set_xlabel("Latitude")
    ax.set_ylabel(f"{title} [{units}]")
    ax.set_xlim(-90, 90)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def amoc_streamfunction(psi, lat, depth, out_path, amoc_max=None, rapid=None,
                        title="Atlantic overturning streamfunction"):
    """Latitude-depth Atlantic overturning streamfunction [Sv].

    psi: (nz, ny) Sv; lat: (ny) deg; depth: (nz) layer-centre depths [m].
    """
    psi = np.asarray(psi)
    fin = psi[np.isfinite(psi)]
    vlim = float(np.nanpercentile(np.abs(fin), 99)) if fin.size else 1.0
    vlim = max(vlim, 1.0)
    fig, ax = plt.subplots(figsize=(COL1 * 1.5, COL1 * 0.95))
    lev = np.linspace(-vlim, vlim, 21)
    cf = ax.contourf(lat, depth, psi, levels=lev, cmap="RdBu_r", extend="both")
    ax.contour(lat, depth, psi, levels=lev[::4], colors="k", linewidths=0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Depth [m]")
    sub = title
    if amoc_max is not None:
        sub += f"\nmax = {amoc_max:.1f} Sv"
        if rapid is not None:
            sub += f"  (RAPID ~ {rapid:.0f} Sv)"
    ax.set_title(sub)
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label("Sv", fontsize=6)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def taylor_diagram(entries, out_path, title="Taylor diagram"):
    """Normalized Taylor diagram.

    entries: list of dicts with keys 'name', 'corr', 'std_ratio'.
    Radius = std_ratio (normalized by obs std), angle = arccos(corr).
    The reference (obs) point sits at (1, 0).
    """
    fig = plt.figure(figsize=(COL1, COL1))
    ax = fig.add_subplot(111, polar=True)
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    # Correlation gridlines.
    corr_ticks = np.array([0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 0.99])
    ax.set_thetagrids(np.degrees(np.arccos(corr_ticks)),
                      labels=[f"{c:g}" for c in corr_ticks])

    rmax = 1.5
    for e in entries:
        corr = e.get("corr", np.nan)
        r = e.get("std_ratio", np.nan)
        if not (np.isfinite(corr) and np.isfinite(r)):
            continue
        theta = np.arccos(np.clip(corr, -1, 1))
        ax.plot(theta, r, "o", ms=4, label=e["name"])
        rmax = max(rmax, r * 1.1)

    # Reference point (obs): corr=1, std_ratio=1.
    ax.plot(0, 1.0, "k*", ms=8, label="Obs (ref)")
    ax.set_rmax(rmax)
    ax.set_rlabel_position(135)
    ax.text(np.radians(45), rmax * 1.18, "correlation", ha="center", fontsize=6)
    ax.set_title(title, pad=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def drift_timeseries(times_years, series, out_path, title="Control-run drift"):
    """Global-mean time series to assess control-run stability/drift.

    series: dict {label: (values, units)}. One small panel per series.
    """
    n = len(series)
    fig, axes = plt.subplots(n, 1, figsize=(COL1, 0.9 * n + 0.5), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, (label, (vals, units)) in zip(axes, series.items()):
        ax.plot(times_years, vals, color="#1b7837", lw=1.0)
        ax.set_ylabel(f"{label}\n[{units}]")
        ax.grid(True, lw=0.3, alpha=0.5)
    axes[-1].set_xlabel("Model year")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
