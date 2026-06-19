"""AMOC hosing / hysteresis experiment (working-ESM P4).

From a steady control state, ramp a subpolar-N-Atlantic freshwater HOSING quasi-
statically UP (0 -> Fmax) then DOWN (Fmax -> 0), holding each level for --hold-years
to (near-)equilibrate, and record AMOC@26.5N. If the up- and down-legs differ over a
finite F window (an open loop with a collapse), that is AMOC HYSTERESIS / a saddle-node
-- the bifurcation that defines tipping. A monotonic, reversible curve means the closure
is (so far) mono-stable (expected until the salt-advection feedback gain exceeds 1).

The SST flux-correction (temperature) does not cancel the hosing (salinity), so the
default restored model is fine. The forced response is differentiable: d(AMOC)/d(F).

    python experiments/run_amoc_hosing.py --ckpt /p/tmp/$USER/dino_control_thc/state_d036500 \
        --fmax 1.0 --nsteps 6 --hold-years 15 --outdir /p/tmp/$USER/amoc_hosing

Writes <outdir>/hysteresis.npz (leg, hosing_sv, amoc_sv) and prints the loop.
Checkpoints the running state per level so a preempted job can be re-pointed.
"""
import argparse
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm.coupler.dino_step import DinoCoupledModel, load_state, save_state  # noqa: E402
from chronos_esm.ocean.diagnostics import compute_amoc  # noqa: E402

DAYS_PER_YEAR = 365


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="steady control checkpoint base (no ext)")
    ap.add_argument("--fmax", type=float, default=1.0, help="max hosing [Sv]")
    ap.add_argument("--nsteps", type=int, default=6, help="levels per leg (up and down)")
    ap.add_argument("--hold-years", type=float, default=15.0, help="years held per level")
    ap.add_argument("--co2", type=float, default=280.0)
    ap.add_argument("--outdir", default="outputs/amoc_hosing")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    model = DinoCoupledModel(ocean_ic="woa")
    cstate = load_state(args.ckpt)
    omask = model.omask

    def amoc_now(cs):
        return float(compute_amoc(cs.ocean, ocean_mask=omask)["upper_cell_26N"])

    up = np.linspace(0.0, args.fmax, args.nsteps)
    legs = [("up", f) for f in up] + [("down", f) for f in up[::-1]]
    n_hold = int(round(args.hold_years * DAYS_PER_YEAR))

    print(f"AMOC hosing sweep from {args.ckpt}: 0->{args.fmax}->0 Sv, {args.nsteps} levels/leg, "
          f"{args.hold_years:.0f} yr/level. Start AMOC = {amoc_now(cstate):+.2f} Sv", flush=True)

    rec_leg, rec_f, rec_amoc = [], [], []
    for i, (leg, f) in enumerate(legs):
        amoc_acc, nacc = 0.0, 0
        for d in range(n_hold):
            cstate = model.step_fast(cstate, co2_ppm=args.co2, hosing_sv=float(f))
            if d >= n_hold - DAYS_PER_YEAR:          # average AMOC over the final year
                amoc_acc += amoc_now(cstate); nacc += 1
        amoc = amoc_acc / max(nacc, 1)
        rec_leg.append(leg); rec_f.append(float(f)); rec_amoc.append(amoc)
        print(f"  [{leg:>4}] hosing {f:5.2f} Sv -> AMOC {amoc:+6.2f} Sv "
              f"(finite {bool(np.isfinite(np.asarray(cstate.ocean.temp)).all())})", flush=True)
        save_state(cstate, os.path.join(args.outdir, f"hose_{i:02d}_{leg}_{f:.2f}"))
        np.savez(os.path.join(args.outdir, "hysteresis.npz"),
                 leg=np.array(rec_leg), hosing_sv=np.array(rec_f), amoc_sv=np.array(rec_amoc))

    a = np.array(rec_amoc); f = np.array(rec_f); lg = np.array(rec_leg)
    up_a = a[lg == "up"]; dn_a = a[lg == "down"][::-1]
    gap = float(np.max(np.abs(up_a - dn_a))) if len(up_a) == len(dn_a) else float("nan")
    print("=" * 60)
    print(f"  AMOC: {a[0]:+.2f} Sv (F=0) -> {a[args.nsteps-1]:+.2f} Sv (F={args.fmax}) "
          f"-> {a[-1]:+.2f} Sv (F=0 back)")
    print(f"  max up-vs-down gap = {gap:.2f} Sv  "
          f"({'HYSTERESIS (tipping)' if gap > 1.0 else 'reversible (mono-stable so far)'})")
    print("=" * 60)


if __name__ == "__main__":
    main_cli()
