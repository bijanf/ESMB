"""Compare Chronos-ESM's idealized DECK runs to the CMIP6 / IPCC-AR6 spread.

Reads the DECK outputs written by run_dino_deck.py (deck_piControl.npz +
deck_abrupt-4xCO2.npz and/or deck_1pctCO2.npz), removes the piControl drift, and
places Chronos-ESM's transient warming and AMOC weakening against the published
CMIP6 / AR6 multi-model ranges.

This is the "check vs CMIP6" step. The CMIP6/AR6 numbers below are the ASSESSED
multi-model RANGES from the literature (cited inline); for a publication, replace
them with output pulled from ESGF for the matching experiments. Chronos-ESM is an
idealized/EMIC-tier model: the question is "does it fall within / near the CMIP6
envelope?", not "does it beat the CMIP6 models". Its sensitivity is a TCR-like
surface-forcing PROXY (no TOA closure -> not a calibrated ECS); see docs/manual
ch20 + appendix E.

    python experiments/compare_cmip6_deck.py --deckdir outputs/deck

Writes docs/figures/cmip6_deck_comparison.pdf (vector) + prints the comparison.
"""

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Published CMIP6 / IPCC-AR6 reference ranges (assessed multi-model spreads).
# ---------------------------------------------------------------------------
CMIP6_REF = {
    # Transient Climate Response [K]. AR6 assessed *likely* range; CMIP6 raw spread
    # is wider. (IPCC AR6 WG1 Ch.7; Meehl et al. 2020, Sci. Adv.)
    "TCR_K": {"low": 1.4, "best": 1.8, "high": 2.2, "source": "IPCC AR6 (likely)"},
    # Equilibrium Climate Sensitivity [K]. AR6 *likely* 2.5-4 (best 3); CMIP6 raw
    # 1.8-5.6. We can only report a PROXY, not ECS -- shown for context only.
    "ECS_K": {"low": 2.5, "best": 3.0, "high": 4.0, "source": "IPCC AR6 (likely)"},
    # AMOC weakening at 2xCO2 under 1pctCO2 [% of initial]. CMIP6 ~ -20 to -40%.
    # (Weijer et al. 2020, GRL, doi:10.1029/2019GL086075)
    "AMOC_weakening_pct_at_2x": {
        "low": -40.0,
        "best": -30.0,
        "high": -20.0,
        "source": "Weijer et al. 2020 (CMIP6)",
    },
}


def _load(deckdir, experiment):
    path = os.path.join(deckdir, f"deck_{experiment}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def _anomaly(forced, control):
    """forced - control on the shared year axis (drift removal)."""
    yc, sc, ac = control["years"], control["sst_C"], control["amoc_Sv"]
    yf = forced["years"]
    sst_ctrl = np.interp(yf, yc, sc)
    amoc_ctrl = np.interp(yf, yc, ac)
    return yf, forced["sst_C"] - sst_ctrl, forced["amoc_Sv"], amoc_ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deckdir", default="outputs/deck")
    ap.add_argument("--out", default="docs/figures/cmip6_deck_comparison.pdf")
    args = ap.parse_args()

    control = _load(args.deckdir, "piControl")
    if control is None:
        sys.exit(
            f"No deck_piControl.npz in {args.deckdir}. Run e.g.:\n"
            "  python experiments/run_dino_deck.py --experiment piControl --ckpt ...\n"
            "  python experiments/run_dino_deck.py --experiment 1pctCO2 --ckpt ...\n"
            "  python experiments/run_dino_deck.py --experiment abrupt-4xCO2 --ckpt ..."
        )

    results = {}

    ramp = _load(args.deckdir, "1pctCO2")
    if ramp is not None:
        yr, dsst, amoc, amoc_ctrl = _anomaly(ramp, control)
        # TCR proxy = warming at the time CO2 first reaches 2x C0 (~yr 70)
        i2x = int(np.argmin(np.abs(ramp["co2_ppm"] - 2.0 * ramp["co2_ppm"][0])))
        results["TCR_proxy_K"] = float(dsst[i2x])
        a0 = amoc[0] if abs(amoc[0]) > 1e-6 else np.nan
        results["AMOC_weakening_pct_at_2x"] = float(100.0 * (amoc[i2x] - a0) / a0)
        print(
            f"1pctCO2: TCR-proxy warming at 2xCO2 (yr {yr[i2x]:.0f}) = "
            f"{results['TCR_proxy_K']:+.2f} K; AMOC "
            f"{amoc[0]:.1f}->{amoc[i2x]:.1f} Sv "
            f"({results['AMOC_weakening_pct_at_2x']:+.0f}%)"
        )

    abrupt = _load(args.deckdir, "abrupt-4xCO2")
    if abrupt is not None:
        yr, dsst, amoc, _ = _anomaly(abrupt, control)
        F = float(abrupt["forcing_Wm2"][-1])
        results["abrupt4x_dSST_K"] = float(dsst[-1])
        results["abrupt4x_K_per_Wm2"] = float(dsst[-1] / max(F, 1e-9))
        print(
            f"abrupt-4xCO2: dSST(end) = {dsst[-1]:+.2f} K at F={F:.1f} W/m2 "
            f"-> {results['abrupt4x_K_per_Wm2']:+.3f} K per W/m2 (surface PROXY, not ECS)"
        )

    _print_placement(results)
    _plot(args, control, ramp, abrupt, results)


def _print_placement(results):
    print("\n--- placement vs CMIP6/AR6 (assessed ranges) ---")
    if "TCR_proxy_K" in results:
        r = CMIP6_REF["TCR_K"]
        inside = r["low"] <= results["TCR_proxy_K"] <= r["high"]
        print(
            f"  TCR proxy {results['TCR_proxy_K']:+.2f} K vs CMIP6/AR6 "
            f"{r['low']}-{r['high']} K (best {r['best']}; {r['source']}) -> "
            f"{'WITHIN' if inside else 'OUTSIDE'} range"
        )
    if "AMOC_weakening_pct_at_2x" in results:
        r = CMIP6_REF["AMOC_weakening_pct_at_2x"]
        inside = r["low"] <= results["AMOC_weakening_pct_at_2x"] <= r["high"]
        print(
            f"  AMOC weakening {results['AMOC_weakening_pct_at_2x']:+.0f}% vs CMIP6 "
            f"{r['low']} to {r['high']}% (best {r['best']}; {r['source']}) -> "
            f"{'WITHIN' if inside else 'OUTSIDE'} range"
        )
    print(
        "  NOTE: replace CMIP6_REF with ESGF-pulled multi-model output for a "
        "publication-grade comparison."
    )


def _plot(args, control, ramp, abrupt, results):
    import matplotlib

    matplotlib.use("pdf")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 6,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.09, 3.0))

    ax = axes[0]
    if ramp is not None:
        yr, dsst, _, _ = _anomaly(ramp, control)
        ax.plot(yr, dsst, label="1pctCO2", color="#c1272d")
    if abrupt is not None:
        yr, dsst, _, _ = _anomaly(abrupt, control)
        ax.plot(yr, dsst, label="abrupt-4xCO2", color="#0000a7")
    ax.set_xlabel("year")
    ax.set_ylabel(r"$\Delta$SST vs piControl [K]")
    ax.set_title("DECK warming (surface proxy)")
    ax.legend(frameon=False)

    ax = axes[1]
    labels, vals, lows, highs = [], [], [], []
    if "TCR_proxy_K" in results:
        r = CMIP6_REF["TCR_K"]
        labels.append("TCR proxy [K]")
        vals.append(results["TCR_proxy_K"])
        lows.append(r["low"])
        highs.append(r["high"])
    for i, (lab, v, lo, hi) in enumerate(zip(labels, vals, lows, highs)):
        ax.fill_between(
            [i - 0.3, i + 0.3],
            lo,
            hi,
            color="0.8",
            label="CMIP6/AR6" if i == 0 else None,
        )
        ax.plot(i, v, "o", color="#c1272d", label="Chronos-ESM" if i == 0 else None)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_title("vs CMIP6/AR6 range")
    if labels:
        ax.legend(frameon=False)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
