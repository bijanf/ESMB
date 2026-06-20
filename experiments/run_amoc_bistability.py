"""AMOC bistability test at FIXED hosing (P4) -- the rigorous fold check.

A quasi-static hosing SWEEP with finite holds measures a RATE-DEPENDENT loop: near the
fold the AMOC collapse timescale is multi-decadal, so 15-yr holds undersample it and a
genuine bistability is indistinguishable from a slow reversible response (see the v2 grid:
the down-leg kept collapsing during the *first* down-leg hold at the same F as the up-leg).

The clean test for true bistability is initial-condition dependence at a SINGLE fixed F:
integrate one run from an ON state and one from an OFF state, both held at the same hosing
for many decades. If they CONVERGE -> monostable (the loop was rate-dependent). If they
stay SEPARATED (on stays strong, off stays collapsed) -> the system is genuinely BISTABLE
at that F, i.e. F sits inside the hysteresis window and a saddle-node fold exists.

The v2 sweep saved both branches as per-level checkpoints (hose_NN_up_F / hose_NN_down_F),
so use the up-leg state at F as the ON IC and the down-leg state at the same F as the OFF
IC. Run this script once per IC (same --hosing-sv, --years), then compare the AMOC(t).

    python experiments/run_amoc_bistability.py --ckpt <on_state_base>  --hosing-sv 0.57 \
        --haline-gain 6 --k-vel 6e-5 --contrast-depth 300 --years 80 --outdir <od>/on
    python experiments/run_amoc_bistability.py --ckpt <off_state_base> --hosing-sv 0.57 \
        --haline-gain 6 --k-vel 6e-5 --contrast-depth 300 --years 80 --outdir <od>/off

Writes <outdir>/amoc_series.npz (year, amoc_sv) + a final-state checkpoint. Auto-resumes
from amoc_series.npz + the latest year checkpoint so a --requeue job continues.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm.coupler.dino_step import DinoCoupledModel, load_state, save_state  # noqa: E402
from chronos_esm.ocean.diagnostics import compute_amoc  # noqa: E402

DAYS_PER_YEAR = 365


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="initial-state checkpoint base (no ext)")
    ap.add_argument("--hosing-sv", type=float, required=True, help="FIXED hosing held [Sv]")
    ap.add_argument("--years", type=float, default=80.0, help="integration length [yr]")
    ap.add_argument("--haline-gain", type=float, default=1.0)
    ap.add_argument("--k-vel", type=float, default=1.0e-4)
    ap.add_argument("--contrast-depth", type=float, default=None)
    ap.add_argument("--co2", type=float, default=280.0)
    ap.add_argument("--log-every", type=float, default=1.0, help="log AMOC every N years")
    ap.add_argument("--ckpt-every", type=float, default=10.0, help="checkpoint every N years")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--fresh", action="store_true", help="ignore existing resume state")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    model = DinoCoupledModel(ocean_ic="woa", thc_haline_gain=args.haline_gain,
                             thc_contrast_depth_m=args.contrast_depth, thc_k_vel=args.k_vel)
    omask = model.omask

    def amoc_now(cs):
        return float(compute_amoc(cs.ocean, ocean_mask=omask)["upper_cell_26N"])

    # AUTO-RESUME: continue from the latest year_* checkpoint + amoc_series.npz.
    yr0, years, amocs = 0, [], []
    series = os.path.join(args.outdir, "amoc_series.npz")
    yr_ckpts = sorted(glob.glob(os.path.join(args.outdir, "year_*_dino.npz")))
    if not args.fresh and yr_ckpts and os.path.exists(series):
        yr0 = max(int(os.path.basename(p).split("_")[1]) for p in yr_ckpts)
        base = glob.glob(os.path.join(args.outdir, f"year_{yr0:03d}_*_dino.npz"))
        base = base[0][: -len("_dino.npz")] if base else \
            os.path.join(args.outdir, f"year_{yr0:03d}")
        cstate = load_state(base)
        d = np.load(series)
        years = [int(x) for x in d["year"] if int(x) <= yr0]
        amocs = [float(x) for x in d["amoc_sv"]][: len(years)]
        print(f"RESUME: year {yr0}; AMOC = {amoc_now(cstate):+.2f} Sv", flush=True)
    else:
        cstate = load_state(args.ckpt)
        print(f"IC {args.ckpt}: AMOC = {amoc_now(cstate):+.2f} Sv; hold at hosing "
              f"{args.hosing_sv} Sv for {args.years:.0f} yr "
              f"(g={args.haline_gain}, k_vel={args.k_vel:.1e}, cd={args.contrast_depth})",
              flush=True)

    n_years = int(round(args.years))
    log_e = max(1, int(round(args.log_every)))
    ck_e = max(1, int(round(args.ckpt_every)))
    for yr in range(yr0 + 1, n_years + 1):
        for _ in range(DAYS_PER_YEAR):
            cstate = model.step_fast(cstate, co2_ppm=args.co2, hosing_sv=float(args.hosing_sv))
        if yr % log_e == 0 or yr == n_years:
            a = amoc_now(cstate)
            years.append(yr); amocs.append(a)
            fin = bool(np.isfinite(np.asarray(cstate.ocean.temp)).all())
            print(f"  yr {yr:3d}/{n_years}: AMOC {a:+6.2f} Sv (finite {fin})", flush=True)
            np.savez(series, year=np.array(years), amoc_sv=np.array(amocs))
        if yr % ck_e == 0 or yr == n_years:
            save_state(cstate, os.path.join(args.outdir, f"year_{yr:03d}"))

    a = np.array(amocs)
    last = a[-min(len(a), 20):]
    print("=" * 60)
    print(f"  IC AMOC {a[0]:+.2f} -> final {a[-1]:+.2f} Sv "
          f"(last-20yr mean {float(last.mean()):+.2f} +/- {float(last.std()):.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main_cli()
