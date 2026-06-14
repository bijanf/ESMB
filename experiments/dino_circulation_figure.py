"""Dashboard figure for the multi-level dinosaur atmosphere after the
eddy-onset fix (near-equilibrium init + rotational seed).

Spins the SST-coupled atmosphere on WOA SST and plots the time-mean circulation
that the dycore now EARNS dynamically: a baroclinic jet with eddy-driven
mid-latitude SURFACE WESTERLIES (the field that was missing), surface winds, and
the ITCZ precipitation. Saves a PNG for the README dashboard.

    python experiments/dino_circulation_figure.py --spinup 90 --avg 30
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chronos_esm import main  # noqa: E402
from chronos_esm.atmos.dino_atmos import DinoAtmosphere  # noqa: E402

LAT_LIN = np.linspace(-90, 90, 48)


def lin_to_gauss(f_lin, lat_g):
    f = np.asarray(f_lin)
    return np.stack([np.interp(lat_g, LAT_LIN, f[:, j]) for j in range(f.shape[1])], axis=0)


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spinup", type=int, default=90)
    ap.add_argument("--avg", type=int, default=30)
    ap.add_argument("--out", default="docs/figures/dino_circulation.png")
    args = ap.parse_args()

    atm = DinoAtmosphere()
    lat_g = atm.lat_deg
    sigma = atm.sigma
    st = main.init_model(ocean_ic="woa")
    sst_g = lin_to_gauss(np.asarray(st.ocean.temp[0]), lat_g)

    state = atm.initial_state(sst_g)
    # spin-up phase (no sampling)
    for _ in range((args.spinup - args.avg) // 5):
        state = atm.step(state, sst_g, n_days=5)
    # averaging phase: sample densely (every 2 d) so transient eddies average out
    accU = accUsfc = accVsfc = accP = None
    n = 0
    for _ in range(args.avg // 2):
        state = atm.step(state, sst_g, n_days=2)
        di = atm.diagnostics(state)
        U = di["u"].mean(axis=1)                      # zonal-mean U (lev,lat)
        accU = U if accU is None else accU + U
        accUsfc = di["u_sfc"] if accUsfc is None else accUsfc + di["u_sfc"]
        accVsfc = di["v_sfc"] if accVsfc is None else accVsfc + di["v_sfc"]
        accP = di["precip"] if accP is None else accP + di["precip"]
        n += 1
    Uzm = accU / n
    usfc = (accUsfc / n).T                                 # (nlat,nlon)
    vsfc = (accVsfc / n).T
    precip = (accP / n).T * 86400.0                        # mm/day

    usfc_zm = usfc.mean(axis=1)
    print("zonal-mean surface U by band [m/s]:")
    for lo, hi in [(-90,-60),(-60,-30),(-30,-10),(-10,10),(10,30),(30,60),(60,90)]:
        m = (lat_g >= lo) & (lat_g < hi)
        print(f"  {lo:+4d}..{hi:+4d}: {usfc_zm[m].mean():+5.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lon = np.linspace(0, 360, atm.nlon, endpoint=False)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))

    vlim = float(np.nanpercentile(np.abs(Uzm), 99)) or 1.0
    cf0 = ax[0].contourf(lat_g, sigma, Uzm, levels=np.linspace(-vlim, vlim, 21), cmap="RdBu_r", extend="both")
    ax[0].contour(lat_g, sigma, Uzm, levels=[0], colors="k", linewidths=0.4)
    ax[0].invert_yaxis(); ax[0].set_xlabel("Latitude"); ax[0].set_ylabel("sigma")
    ax[0].set_title("Zonal-mean U [m/s] (eddy-driven jet)")
    fig.colorbar(cf0, ax=ax[0], shrink=0.85)

    uv = float(np.nanpercentile(np.abs(usfc), 99)) or 1.0
    cf1 = ax[1].pcolormesh(lon, lat_g, usfc, cmap="RdBu_r", vmin=-uv, vmax=uv)
    ax[1].set_title("Surface zonal wind u_sfc [m/s]"); ax[1].set_xlabel("Longitude"); ax[1].set_ylabel("Latitude")
    fig.colorbar(cf1, ax=ax[1], shrink=0.85)

    cf2 = ax[2].pcolormesh(lon, lat_g, precip, cmap="YlGnBu", vmin=0, vmax=float(np.nanpercentile(precip, 99)))
    ax[2].set_title("Precipitation [mm/day] (ITCZ)"); ax[2].set_xlabel("Longitude"); ax[2].set_ylabel("Latitude")
    fig.colorbar(cf2, ax=ax[2], shrink=0.85)

    fig.suptitle(f"Multi-level dinosaur atmosphere on WOA SST (mean of last {args.avg} d of {args.spinup} d)", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main_cli()
