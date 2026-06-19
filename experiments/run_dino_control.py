"""Coupled control run on the LIBRARY multi-level stepper (working-ESM P1).

Unlike experiments/run_dino_coupled.py (which inlines the coupling as numpy in a
Python loop), this drives chronos_esm.coupler.dino_step.DinoCoupledModel -- the
single differentiable coupling step (dino atmosphere + ocean + land + Semtner sea
ice) -- and uses its bit-exact checkpoint/resume. This is the unified code path the
forcing-response / tipping / paleo work builds on.

    # 5-year control, checkpoint yearly:
    python experiments/run_dino_control.py --years 5
    # smoke test: 4 days, checkpoint every 2:
    python experiments/run_dino_control.py --days 4 --ckpt-every-days 2
    # resume from day 2 and continue toward the target:
    python experiments/run_dino_control.py --days 4 --resume 2

Checkpoints: <outdir>/state_d<DAY>.npz (+ _dino.npz). Score/inspect via
DinoCoupledModel.diagnostics_lin or a later dashboard export.
"""
import argparse
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm.coupler.dino_step import (DinoCoupledModel, save_state,  # noqa: E402
                                           load_state)

DAYS_PER_YEAR = 365


def _ckpt_base(outdir, day):
    return os.path.join(outdir, f"state_d{day:06d}")


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--days", type=int, default=0, help="total days (overrides --years)")
    ap.add_argument("--interval", type=float, default=1.0, help="coupling interval [days]")
    ap.add_argument("--ckpt-every-days", type=int, default=DAYS_PER_YEAR)
    ap.add_argument("--outdir", default="outputs/dino_control_lib")
    ap.add_argument("--resume", type=int, default=None, help="resume from an absolute day")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    end_day = args.days if args.days > 0 else int(round(args.years * DAYS_PER_YEAR))

    model = DinoCoupledModel(ocean_ic="woa")
    omask = model.omask

    def sst_mean_C(cstate):
        sst = cstate.ocean.temp[0]
        return float(jnp.sum(jnp.where(omask, sst, 0.0)) / jnp.sum(omask)) - 273.15

    if args.resume is not None:
        cstate = load_state(_ckpt_base(args.outdir, args.resume))
        day = args.resume
        print(f"Resumed from day {day} ({day / DAYS_PER_YEAR:.2f} yr)", flush=True)
    else:
        cstate = model.init_state()
        day = 0

    if day >= end_day:
        print(f"start day {day} >= target {end_day}; nothing to do.")
        return

    print(f"Library dino control run: days {day}->{end_day} (interval {args.interval}d, "
          f"checkpoint every {args.ckpt_every_days}d)", flush=True)

    n_intervals = int(round((end_day - day) / args.interval))
    for it in range(1, n_intervals + 1):
        cstate = model.step(cstate, interval=args.interval)
        day = int(round(cstate.day))
        finite = bool(np.isfinite(np.asarray(cstate.ocean.temp)).all()
                      and np.isfinite(np.asarray(cstate.atmos.vorticity)).all())
        if (day % 30 == 0) or (it == n_intervals) or (not finite):
            ice_area = float(jnp.sum(jnp.where(omask, cstate.ice.concentration, 0.0)))
            print(f"  day {day:6d} ({day / DAYS_PER_YEAR:6.2f} yr): "
                  f"SST {sst_mean_C(cstate):5.2f}C  ice-area {ice_area:8.0f}  "
                  f"|curr|max {float(jnp.abs(cstate.ocean.u).max()):.3f}  finite {finite}",
                  flush=True)
        if not finite:
            save_state(cstate, _ckpt_base(args.outdir, day) + "_NAN")
            raise FloatingPointError(f"non-finite state at day {day}; dump written")
        if (day % args.ckpt_every_days == 0) or (it == n_intervals):
            save_state(cstate, _ckpt_base(args.outdir, day))
            print(f"  [checkpoint] day {day} -> {_ckpt_base(args.outdir, day)}.npz", flush=True)

    print(f"done: ran to day {day} ({day / DAYS_PER_YEAR:.2f} yr).", flush=True)


if __name__ == "__main__":
    main_cli()
