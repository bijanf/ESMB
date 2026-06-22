"""Mid-Holocene (6 ka) vs present-day (PI) figure — the P5 deliverable.

Reads the two seasonal-climatology files written by run_paleo_midholocene.py
(clim_pi.npz, clim_6ka.npz: raw JJA/DJF sums + counts) and makes the headline
mid-Holocene result:

  (a) JJA precipitation anomaly 6 ka - PI [mm/day] with PI JJA precip contours
      -> the enhanced / northward-shifted NH summer monsoon ("Green Sahara");
  (b) zonal-mean JJA precip, PI vs 6 ka -> the ITCZ / monsoon-rain northward shift;
  (c) JJA 2 m-temperature anomaly 6 ka - PI [K] -> NH summer warming (esp. land),
      the orbital-insolation fingerprint.

Vector PDF, Nature-style (sans-serif <=7 pt, RGB, 2-column 180 mm). Usage:
    python experiments/plot_paleo_midholocene.py --pi outputs/paleo_pi/clim_pi.npz \
        --ho outputs/paleo_6ka/clim_6ka.npz --out docs/figures/paleo_midholocene.pdf
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chronos_esm.config import OCEAN_GRID  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})

NLAT, NLON = OCEAN_GRID.nlat, OCEAN_GRID.nlon
LAT = np.linspace(-90, 90, NLAT)
LON = np.linspace(0, 360, NLON, endpoint=False)
SEC_PER_DAY = 86400.0


def _mean(d, season, field):
    n = max(int(d[f"n_{season}"]), 1)
    return np.asarray(d[f"{season}_{field}"]) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi", required=True, help="clim_pi.npz")
    ap.add_argument("--ho", required=True, help="clim_6ka.npz")
    ap.add_argument("--out", default="docs/figures/paleo_midholocene.pdf")
    args = ap.parse_args()

    pi, ho = np.load(args.pi), np.load(args.ho)
    npi, nho = int(pi["n_jja"]), int(ho["n_jja"])
    pr_pi = _mean(pi, "jja", "precip") * SEC_PER_DAY     # mm/day
    pr_ho = _mean(ho, "jja", "precip") * SEC_PER_DAY
    dpr = pr_ho - pr_pi
    dt2 = _mean(ho, "jja", "t2m") - _mean(pi, "jja", "t2m")   # K

    fig = plt.figure(figsize=(7.09, 2.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.8, 1.25], wspace=0.32)

    # (a) JJA precip anomaly map + PI contours
    axa = fig.add_subplot(gs[0, 0])
    vlim = float(np.nanpercentile(np.abs(dpr), 98)) or 1.0
    im = axa.pcolormesh(LON, LAT, dpr, cmap="BrBG", vmin=-vlim, vmax=vlim, shading="auto")
    cs = axa.contour(LON, LAT, pr_pi, levels=[2, 5, 8], colors="k", linewidths=0.4, alpha=0.6)
    axa.clabel(cs, fmt="%d", fontsize=4)
    axa.set_title("(a) JJA precip: 6 ka - PI", loc="left")
    axa.set_xlabel("longitude"); axa.set_ylabel("latitude")
    axa.set_xticks([0, 90, 180, 270, 360]); axa.set_yticks([-60, -30, 0, 30, 60])
    cb = fig.colorbar(im, ax=axa, fraction=0.046, pad=0.03); cb.set_label("mm day$^{-1}$")

    # (b) zonal-mean JJA precip
    axb = fig.add_subplot(gs[0, 1])
    axb.plot(pr_pi.mean(1), LAT, "0.3", lw=1.0, label="PI")
    axb.plot(pr_ho.mean(1), LAT, "C1", lw=1.0, label="6 ka")
    axb.axhline(0, color="0.7", lw=0.4)
    axb.set_title("(b) zonal-mean JJA precip", loc="left")
    axb.set_xlabel("mm day$^{-1}$"); axb.set_ylabel("latitude")
    axb.set_yticks([-60, -30, 0, 30, 60]); axb.legend(frameon=False, loc="upper right")

    # (c) JJA 2 m-temperature anomaly
    axc = fig.add_subplot(gs[0, 2])
    tlim = float(np.nanpercentile(np.abs(dt2), 98)) or 1.0
    im2 = axc.pcolormesh(LON, LAT, dt2, cmap="RdBu_r", vmin=-tlim, vmax=tlim, shading="auto")
    axc.set_title("(c) JJA 2 m T: 6 ka - PI", loc="left")
    axc.set_xlabel("longitude"); axc.set_ylabel("latitude")
    axc.set_xticks([0, 90, 180, 270, 360]); axc.set_yticks([-60, -30, 0, 30, 60])
    cb2 = fig.colorbar(im2, ax=axc, fraction=0.046, pad=0.03); cb2.set_label("K")

    fig.suptitle(f"Mid-Holocene 6 ka vs PI (free seasonal coupled run; "
                 f"JJA samples PI={npi}, 6 ka={nho})", fontsize=7, y=1.02)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")

    # quick numeric summary for the caption / sanity
    nh_monsoon = (LAT[:, None] >= 5) & (LAT[:, None] <= 35)
    print(f"NH (5-35N) JJA precip:  PI {np.mean(pr_pi[nh_monsoon[:,0]]):.2f} -> "
          f"6 ka {np.mean(pr_ho[nh_monsoon[:,0]]):.2f} mm/day "
          f"(delta {np.mean(dpr[nh_monsoon[:,0]]):+.2f})")
    # ITCZ latitude = precip-weighted mean lat in the tropics
    trop = np.abs(LAT) <= 30
    def itcz(pr):
        w = pr[trop].mean(1); return float(np.sum(LAT[trop] * w) / np.sum(w))
    print(f"tropical JJA ITCZ latitude: PI {itcz(pr_pi):+.1f} -> 6 ka {itcz(pr_ho):+.1f} deg")


if __name__ == "__main__":
    main()
