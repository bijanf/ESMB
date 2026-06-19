"""Forced CO2 experiment on the freed (q-flux) coupled model -- working-ESM P2.

Given a converged CONTROL checkpoint and its frozen q-flux (state_d<DAY>.npz +
state_d<DAY>_qflux.npy from run_dino_control.py), this builds the FREE model
(q-flux + weak long-tau anomaly restoring, so SST can respond to forcing) and runs
TWO branches from the same control state:

  * baseline  : co2 = control ppm (default 280)  -> residual drift reference
  * forced    : co2 = forced ppm  (default 560)  -> 2xCO2

The CO2 warming is the DIFFERENCE  dSST(t) = forced - baseline  (drift cancels).
This is an honest TCR-like forcing-vs-warming PROXY -- there is no TOA energy closure,
so it is NOT an equilibrium ECS.

    python experiments/run_dino_co2.py --ckpt /p/tmp/.../state_d036500 \
        --co2 560 --years 50 --outdir outputs/co2_2x

Writes <outdir>/warming.npz (years, sst_baseline, sst_forced, dSST) and prints the
forcing-vs-warming slope.
"""
import argparse
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm.coupler.dino_step import DinoCoupledModel, load_state  # noqa: E402
from chronos_esm.coupler.dino_coupling import co2_forcing_wm2  # noqa: E402

DAYS_PER_YEAR = 365


def _run_branch(model, cstate, co2_ppm, end_day, interval, label):
    """Step the free model to end_day at fixed co2_ppm; return [(year, ocean-mean SST C)]."""
    omask = model.omask
    curve = []

    def sst_mean_C(cs):
        sst = cs.ocean.temp[0]
        return float(jnp.sum(jnp.where(omask, sst, 0.0)) / jnp.sum(omask)) - 273.15

    day = int(round(cstate.day))
    start = day
    curve.append((day / DAYS_PER_YEAR, sst_mean_C(cstate)))
    n = int(round((end_day - day) / interval))
    for it in range(1, n + 1):
        cstate = model.step(cstate, interval=interval, co2_ppm=co2_ppm)
        day = int(round(cstate.day))
        if (day - start) % 365 == 0 or it == n:
            t, s = day / DAYS_PER_YEAR, sst_mean_C(cstate)
            curve.append((t, s))
            fin = bool(np.isfinite(np.asarray(cstate.ocean.temp)).all())
            print(f"  [{label}] yr {t - start/DAYS_PER_YEAR:6.2f}  SST {s:6.3f}C  finite {fin}",
                  flush=True)
            if not fin:
                raise FloatingPointError(f"[{label}] non-finite at day {day}")
    return np.array(curve)


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="control checkpoint base (no extension)")
    ap.add_argument("--qflux", default=None, help="q-flux .npy (default <ckpt>_qflux.npy)")
    ap.add_argument("--co2", type=float, default=560.0, help="forced CO2 [ppm]")
    ap.add_argument("--control-co2", type=float, default=280.0, help="baseline CO2 [ppm]")
    ap.add_argument("--years", type=float, default=50.0)
    ap.add_argument("--days", type=int, default=0)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--restore-tau-days", type=float, default=3650.0,
                    help="weak anomaly-restoring timescale in FREE mode")
    ap.add_argument("--outdir", default="outputs/co2_run")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    qpath = args.qflux or (args.ckpt + "_qflux.npy")
    q_flux = jnp.asarray(np.load(qpath))
    print(f"q-flux from {qpath}: mean {float(jnp.mean(q_flux)):.2f} W/m2", flush=True)

    model = DinoCoupledModel(q_flux=q_flux, restore_tau_days=args.restore_tau_days)
    cstate0 = load_state(args.ckpt)
    start_day = int(round(cstate0.day))
    end_day = (start_day + args.days) if args.days > 0 else start_day + int(round(args.years * DAYS_PER_YEAR))

    print(f"FREE-mode CO2 experiment from day {start_day}: baseline {args.control_co2} ppm "
          f"vs forced {args.co2} ppm (F={float(co2_forcing_wm2(args.co2)):.2f} W/m2 vs "
          f"{float(co2_forcing_wm2(args.control_co2)):.2f}); tau_weak={args.restore_tau_days}d", flush=True)

    base = _run_branch(model, cstate0, args.control_co2, end_day, args.interval, "base")
    forced = _run_branch(model, cstate0, args.co2, end_day, args.interval, "2xCO2")

    yrs = base[:, 0] - base[0, 0]
    dSST = forced[:, 1] - base[:, 1]
    np.savez(os.path.join(args.outdir, "warming.npz"),
             years=yrs, sst_baseline=base[:, 1], sst_forced=forced[:, 1], dSST=dSST)
    F = float(co2_forcing_wm2(args.co2) - co2_forcing_wm2(args.control_co2))
    print("=" * 60)
    print(f"  forcing F = {F:.2f} W/m2 ({args.control_co2:.0f}->{args.co2:.0f} ppm)")
    print(f"  warming dSST(end) = {dSST[-1]:+.3f} K   (TCR-like proxy, NOT equilibrium ECS)")
    print(f"  -> {dSST[-1] / max(F, 1e-9):+.3f} K per W/m2")
    print("=" * 60)


if __name__ == "__main__":
    main_cli()
