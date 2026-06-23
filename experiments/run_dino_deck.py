"""CMIP-DECK-shaped idealized experiments on the freed (q-flux) coupled model.

This runs ONE DECK-shaped branch of the differentiable coupled model and records the
annual SST, AMOC, and global-mean ocean temperature so the result can be compared to
the CMIP6 multi-model spread (experiments/compare_cmip6_deck.py).

These are CMIP-*shaped* idealized experiments, NOT CMIP-class output: the response is a
TCR-like surface-forcing PROXY (no TOA energy closure -> no calibrated ECS), the grid is
T31, and the forcing is CO2-only. See docs/manual ch20 (Toward CMIP7 Participation) and
appendix E (Model Limitations).

Experiments (CO2 schedule, C0 = control ppm, default 280):
  * piControl     : C = C0                       (drift/forcing-free reference)
  * abrupt-2xCO2  : C = 2*C0                      (= 560 ppm)
  * abrupt-4xCO2  : C = 4*C0                      (= 1120 ppm, the DECK sensitivity run)
  * 1pctCO2       : C = C0 * 1.01**year, capped at 4*C0 (compound 1%/yr ramp)

Like run_dino_co2.py it starts from a converged CONTROL checkpoint and its frozen q-flux
(state_d<DAY>.npz + state_d<DAY>_qflux.npy from run_dino_control.py) and runs the FREE
model (q-flux + weak long-tau anomaly restoring) so SST can respond to the forcing.

    python experiments/run_dino_deck.py --experiment abrupt-4xCO2 \
        --ckpt /p/tmp/$USER/dino_control/state_d036500 --years 150 \
        --outdir outputs/deck

    # smoke test the plumbing with no control checkpoint (NOT a science run):
    python experiments/run_dino_deck.py --experiment abrupt-4xCO2 --fresh --days 4

Writes <outdir>/deck_<experiment>.npz: years, sst_C, amoc_Sv, tglobal_C, co2_ppm,
forcing_Wm2 (annual). Run piControl + the forced branch(es), then compare.
"""

import argparse
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm.coupler.dino_coupling import co2_forcing_wm2  # noqa: E402
from chronos_esm.coupler.dino_step import DinoCoupledModel, load_state  # noqa: E402
from chronos_esm.ocean.diagnostics import compute_amoc  # noqa: E402

DAYS_PER_YEAR = 365
EXPERIMENTS = ("piControl", "abrupt-2xCO2", "abrupt-4xCO2", "1pctCO2")


def co2_for_experiment(experiment, year, control_co2=280.0):
    """CO2 [ppm] for a DECK experiment at `year` elapsed since its start.

    Pure function (no model state) so it can be unit-tested without JAX/data.
    """
    if experiment == "piControl":
        return control_co2
    if experiment == "abrupt-2xCO2":
        return 2.0 * control_co2
    if experiment == "abrupt-4xCO2":
        return 4.0 * control_co2
    if experiment == "1pctCO2":
        # compound 1%/yr, capped at 4x (the standard DECK 1pctCO2 length is ~140 yr,
        # at which point the compounded concentration reaches quadrupling).
        return min(control_co2 * (1.01**year), 4.0 * control_co2)
    raise ValueError(f"unknown experiment {experiment!r}; choose from {EXPERIMENTS}")


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=EXPERIMENTS)
    ap.add_argument(
        "--ckpt", default=None, help="control checkpoint base (no extension)"
    )
    ap.add_argument(
        "--qflux", default=None, help="q-flux .npy (default <ckpt>_qflux.npy)"
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="init from WOA in restoring mode (NO q-flux) -- smoke test only",
    )
    ap.add_argument("--control-co2", type=float, default=280.0)
    ap.add_argument("--years", type=float, default=150.0)
    ap.add_argument(
        "--days", type=int, default=0, help="total days (overrides --years)"
    )
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument(
        "--restore-tau-days",
        type=float,
        default=3650.0,
        help="weak anomaly-restoring timescale in FREE mode",
    )
    ap.add_argument("--outdir", default="outputs/deck")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.fresh:
        print(
            "FRESH mode: WOA init, restoring (NO q-flux). Plumbing smoke test, "
            "NOT a science run.",
            flush=True,
        )
        model = DinoCoupledModel(ocean_ic="woa", interval=args.interval)
        cstate = model.init_state()
    else:
        if args.ckpt is None:
            ap.error("--ckpt is required unless --fresh is given")
        qpath = args.qflux or (args.ckpt + "_qflux.npy")
        q_flux = jnp.asarray(np.load(qpath))
        print(
            f"q-flux from {qpath}: mean {float(jnp.mean(q_flux)):.2f} W/m2", flush=True
        )
        model = DinoCoupledModel(
            q_flux=q_flux,
            restore_tau_days=args.restore_tau_days,
            interval=args.interval,
        )
        cstate = load_state(args.ckpt)

    omask = model.omask

    def sst_mean_C(cs):
        sst = cs.ocean.temp[0]
        return float(jnp.sum(jnp.where(omask, sst, 0.0)) / jnp.sum(omask)) - 273.15

    def tglobal_C(cs):
        t = cs.ocean.temp
        m = model.ocean_mask_3d
        return float(jnp.sum(jnp.where(m > 0.5, t, 0.0)) / jnp.sum(m > 0.5)) - 273.15

    def amoc_Sv(cs):
        return float(
            compute_amoc(cs.ocean, ocean_mask=omask, ocean_mask_3d=model.ocean_mask_3d)[
                "upper_cell_26N"
            ]
        )

    start_day = int(round(cstate.day))
    end_day = (
        (start_day + args.days)
        if args.days > 0
        else start_day + int(round(args.years * DAYS_PER_YEAR))
    )
    n = int(round((end_day - start_day) / args.interval))

    print(
        f"DECK '{args.experiment}' from day {start_day} -> {end_day} "
        f"(C0={args.control_co2:.0f} ppm, interval {args.interval}d)",
        flush=True,
    )

    rows = []  # (year, sst_C, amoc_Sv, tglobal_C, co2_ppm, forcing_Wm2)

    def record(cs):
        yr = (int(round(cs.day)) - start_day) / DAYS_PER_YEAR
        co2 = co2_for_experiment(args.experiment, yr, args.control_co2)
        F = float(co2_forcing_wm2(co2) - co2_forcing_wm2(args.control_co2))
        rows.append((yr, sst_mean_C(cs), amoc_Sv(cs), tglobal_C(cs), co2, F))
        return co2

    record(cstate)
    for it in range(1, n + 1):
        yr = (int(round(cstate.day)) - start_day) / DAYS_PER_YEAR
        co2 = co2_for_experiment(args.experiment, yr, args.control_co2)
        cstate = model.step_fast(cstate, co2_ppm=co2)
        day = int(round(cstate.day))
        if (day - start_day) % DAYS_PER_YEAR == 0 or it == n:
            co2_now = record(cstate)
            r = rows[-1]
            finite = bool(np.isfinite(np.asarray(cstate.ocean.temp)).all())
            print(
                f"  yr {r[0]:6.1f}: CO2 {co2_now:7.1f}ppm (F {r[5]:+5.2f})  "
                f"SST {r[1]:6.3f}C  AMOC {r[2]:5.1f}Sv  finite {finite}",
                flush=True,
            )
            if not finite:
                raise FloatingPointError(f"non-finite at day {day}")

    arr = np.array(rows)
    out = os.path.join(args.outdir, f"deck_{args.experiment}.npz")
    np.savez(
        out,
        years=arr[:, 0],
        sst_C=arr[:, 1],
        amoc_Sv=arr[:, 2],
        tglobal_C=arr[:, 3],
        co2_ppm=arr[:, 4],
        forcing_Wm2=arr[:, 5],
        experiment=args.experiment,
    )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main_cli()
