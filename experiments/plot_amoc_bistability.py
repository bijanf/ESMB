"""AMOC bistability / hysteresis figure (P4) from the fixed-F on/off-IC experiment.

Reads the run_amoc_bistability.py series (one dir per F per branch) and plots the
bifurcation diagram: equilibrated AMOC vs hosing for the ON-IC and OFF-IC branches. A
persistent gap between the branches over the F window = a saddle-node hysteresis window
= genuine AMOC bistability. A second panel shows the IC-dependent trajectories at one F.

Nature-family spec (vector PDF, Helvetica/Arial, RGB, <=7 pt, 1-col 88 mm).

    python experiments/plot_amoc_bistability.py --root /tmp/amoc_bist \
        --pairs 0.46 0.57 0.69 --traj 0.57 --out docs/figures/amoc_bistability.pdf
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _load(root, F, br):
    for cand in (os.path.join(root, f"F{F}_{br}.npz"),
                 os.path.join(root, f"*F{F}_{br}*", "amoc_series.npz")):
        hits = glob.glob(cand)
        if hits:
            d = np.load(hits[0])
            return d["year"], d["amoc_sv"]
    raise FileNotFoundError(f"no series for F={F} {br} under {root}")


def _stats(year, amoc, eq_year=40):
    m = year >= eq_year
    a = amoc[m]
    return a.mean(), a.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/amoc_bist")
    ap.add_argument("--pairs", nargs="+", default=["0.46", "0.57", "0.69"])
    ap.add_argument("--traj", default="0.57", help="F for the trajectory panel")
    ap.add_argument("--eq-year", type=float, default=40.0)
    ap.add_argument("--config", default="k_vel=6e-5, haline_gain=6, contrast_depth=300 m")
    ap.add_argument("--out", default="docs/figures/amoc_bistability.pdf")
    args = ap.parse_args()

    Fs = [float(f) for f in args.pairs]
    on_m, on_s, off_m, off_s = [], [], [], []
    for f in args.pairs:
        ym, am = _load(args.root, f, "on"); m, s = _stats(ym, am, args.eq_year)
        on_m.append(m); on_s.append(s)
        yf, af = _load(args.root, f, "off"); m, s = _stats(yf, af, args.eq_year)
        off_m.append(m); off_s.append(s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.46, 1.8))

    # (a) bifurcation diagram
    ax1.errorbar(Fs, on_m, yerr=on_s, fmt="o-", color="#1f77b4", capsize=2, lw=1,
                 ms=3, label="from ON state")
    ax1.errorbar(Fs, off_m, yerr=off_s, fmt="s--", color="#d62728", capsize=2, lw=1,
                 ms=3, label="from OFF state")
    ax1.fill_between(Fs, off_m, on_m, color="0.85", zorder=0)
    ax1.set_xlabel("subpolar hosing $F$ (Sv)")
    ax1.set_ylabel("AMOC @ 26.5°N (Sv)")
    ax1.set_title("(a) bistable window", loc="left")
    ax1.legend(frameon=False, loc="upper right")
    ax1.set_ylim(0, None)

    # (b) IC-dependent trajectories at one F
    ym, am = _load(args.root, args.traj, "on")
    yf, af = _load(args.root, args.traj, "off")
    ax2.plot(ym, am, color="#1f77b4", lw=0.6, alpha=0.5)
    ax2.plot(yf, af, color="#d62728", lw=0.6, alpha=0.5)
    # running means
    def rm(x, k=10):
        return np.convolve(x, np.ones(k) / k, mode="valid")
    ax2.plot(ym[9:], rm(am), color="#1f77b4", lw=1.4, label="ON IC")
    ax2.plot(yf[9:], rm(af), color="#d62728", lw=1.4, label="OFF IC")
    ax2.set_xlabel("year")
    ax2.set_ylabel("AMOC @ 26.5°N (Sv)")
    ax2.set_title(f"(b) $F$ = {args.traj} Sv", loc="left")
    ax2.legend(frameon=False, loc="upper right")
    ax2.set_ylim(0, None)

    fig.suptitle(f"AMOC bistability ({args.config})", fontsize=7, y=1.04)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")
    print("branch means (Sv):")
    for f, om, os_, fm, fs in zip(args.pairs, on_m, on_s, off_m, off_s):
        print(f"  F={f}: ON {om:5.1f}±{os_:4.1f}   OFF {fm:5.1f}±{fs:4.1f}   gap {om-fm:+.1f}")


if __name__ == "__main__":
    main()
