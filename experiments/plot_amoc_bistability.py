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


def _scan(root, br):
    """All F (sorted) for which a series exists for branch br, with (mean,std)."""
    out = {}
    for p in sorted(glob.glob(os.path.join(root, f"F*_{br}.npz"))):
        F = float(os.path.basename(p).split("_")[0][1:])
        d = np.load(p)
        out[F] = _stats(d["year"], d["amoc_sv"])
    return dict(sorted(out.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/tmp/amoc_bist")
    ap.add_argument("--traj", default="0.57", help="F for the trajectory panel")
    ap.add_argument("--eq-year", type=float, default=40.0)
    ap.add_argument("--collapsed-below", type=float, default=11.0,
                    help="branch AMOC below this = OFF/collapsed state (for window edges)")
    ap.add_argument("--config", default="k_vel=6e-5, haline_gain=6, contrast_depth=300 m")
    ap.add_argument("--out", default="docs/figures/amoc_bistability.pdf")
    args = ap.parse_args()

    on = _scan(args.root, "on")     # {F: (mean,std,neff)} -- on-IC integrations
    off = _scan(args.root, "off")   # off-IC integrations

    # window edges: F_lower = highest off-IC F that RECOVERED (mean high); F_upper =
    # lowest on-IC F that COLLAPSED (mean low). Bistable F have both branches separated.
    thr = args.collapsed_below
    off_low = [F for F, v in off.items() if v[0] < thr + 4]    # off stays low -> in window
    on_hi = [F for F, v in on.items() if v[0] > thr]           # on stays up -> in window
    f_lower = min(off_low) if off_low else min(on)             # window opens
    f_upper = max(on_hi) if on_hi else max(off)                # window closes (on collapses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.46, 1.8))

    # (a) bifurcation diagram -- full window
    Fon = list(on); Foff = list(off)
    ax1.axvspan(f_lower, f_upper, color="0.88", zorder=0, label="bistable window")
    ax1.errorbar(Fon, [on[f][0] for f in Fon], yerr=[on[f][1] for f in Fon],
                 fmt="o-", color="#1f77b4", capsize=2, lw=1, ms=3, label="from ON state")
    ax1.errorbar(Foff, [off[f][0] for f in Foff], yerr=[off[f][1] for f in Foff],
                 fmt="s--", color="#d62728", capsize=2, lw=1, ms=3, label="from OFF state")
    ax1.set_xlabel("subpolar hosing $F$ (Sv)")
    ax1.set_ylabel("AMOC @ 26.5°N (Sv)")
    ax1.set_title(f"(a) hysteresis window ≈ [{f_lower:.2f}, {f_upper:.2f}] Sv", loc="left",
                  fontsize=6)
    ax1.legend(frameon=False, loc="upper right", fontsize=5)
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
    print(f"window ~ [{f_lower:.2f}, {f_upper:.2f}] Sv")
    for f in sorted(set(on) | set(off)):
        os_ = f"{on[f][0]:5.1f}±{on[f][1]:4.1f}" if f in on else "    -     "
        fs_ = f"{off[f][0]:5.1f}±{off[f][1]:4.1f}" if f in off else "    -     "
        print(f"  F={f:.2f}: ON {os_}   OFF {fs_}")


if __name__ == "__main__":
    main()
