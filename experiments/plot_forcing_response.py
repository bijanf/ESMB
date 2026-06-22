"""Forcing-response figure (P2): transient surface warming vs CO2 radiative forcing.

Plots the abrupt-CO2 DECK-style experiments (Chapter 14): global-mean surface warming dSST
against the Myhre forcing F = 5.35 ln(C/280). A reference line through the 2xCO2 point shows
the (sub-)linearity of the response. TCR-like proxy, NOT equilibrium ECS.

    python experiments/plot_forcing_response.py --out docs/figures/forcing_response.pdf
"""
import argparse

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 6, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 6,
    "savefig.dpi": 300, "pdf.fonttype": 42,
})

# (CO2 ppm, forcing F [W/m2], dSST [K]) from the abrupt-CO2 runs (run_dino_co2.py).
POINTS = [(280, 0.0, 0.0), (560, 3.71, 1.58), (1120, 7.42, 2.35)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/forcing_response.pdf")
    args = ap.parse_args()

    F = np.array([p[1] for p in POINTS])
    dT = np.array([p[2] for p in POINTS])
    labels = ["control", r"$2\times$CO$_2$", r"$4\times$CO$_2$"]

    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    # reference: linear response calibrated on the 2xCO2 sensitivity
    s2 = dT[1] / F[1]
    ax.plot([0, F[-1]], [0, s2 * F[-1]], "--", color="0.6", lw=1,
            label=f"linear @ {s2:.2f} K/(W m$^{{-2}}$)")
    ax.plot(F, dT, "o-", color="#b30000", lw=1.4, ms=5)
    for f, t, lab in zip(F, dT, labels):
        ax.annotate(lab, (f, t), textcoords="offset points", xytext=(6, -2), fontsize=6)
    ax.set_xlabel(r"CO$_2$ radiative forcing $F$ (W m$^{-2}$)")
    ax.set_ylabel(r"global-mean surface warming $\Delta$SST (K)")
    ax.set_title("Transient forcing response (TCR-like proxy)", loc="left")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, None); ax.set_ylim(0, None)
    ax.grid(alpha=0.25)
    fig.savefig(args.out, bbox_inches="tight")
    if args.out.endswith(".pdf"):
        fig.savefig(args.out[:-4] + ".png", bbox_inches="tight")
    print(f"wrote {args.out}; sensitivities (K per W/m2):",
          ", ".join(f"{lab}={t/f:.3f}" for f, t, lab in zip(F[1:], dT[1:], labels[1:])))


if __name__ == "__main__":
    main()
