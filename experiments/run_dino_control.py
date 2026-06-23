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

from chronos_esm import orbital  # noqa: E402
from chronos_esm.config import OCEAN_GRID  # noqa: E402
from chronos_esm.coupler.dino_step import save_state  # noqa: E402
from chronos_esm.coupler.dino_step import (
    DinoCoupledModel,
    load_state,
)
from chronos_esm.ocean import flux_correction  # noqa: E402
from chronos_esm.ocean.diagnostics import compute_amoc  # noqa: E402

DAYS_PER_YEAR = 365


def _ckpt_base(outdir, day):
    return os.path.join(outdir, f"state_d{day:06d}")


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument(
        "--days", type=int, default=0, help="total days (overrides --years)"
    )
    ap.add_argument(
        "--interval", type=float, default=1.0, help="coupling interval [days]"
    )
    ap.add_argument("--ckpt-every-days", type=int, default=DAYS_PER_YEAR)
    ap.add_argument("--outdir", default="outputs/dino_control_lib")
    ap.add_argument(
        "--resume", type=int, default=None, help="resume from an absolute day"
    )
    ap.add_argument(
        "--prognostic",
        action="store_true",
        help="P3/S5: use the prognostic baroclinic momentum ocean core (+rigid-lid "
        "projection); also sets thc_k_vel=0 unless --keep-thc",
    )
    ap.add_argument(
        "--mom-drag-days",
        type=float,
        default=30.0,
        help="linear momentum drag timescale [days] for --prognostic",
    )
    ap.add_argument(
        "--prognostic-spherical",
        action="store_true",
        help="P3/S5d: use the NEW spherical prognostic ocean core (implicitly-viscous "
        "no-slip momentum + Munk barotropic streamfunction + GM, mass-conserving); "
        "sets thc_k_vel=0 unless --keep-thc, and a Munk-resolving Ah (--ocean-ah)",
    )
    ap.add_argument(
        "--ocean-ah",
        type=float,
        default=5.0e6,
        help="lateral viscosity Ah [m^2/s] for --prognostic-spherical (Munk-resolving "
        "at T31 ~ 5e6)",
    )
    ap.add_argument(
        "--keep-thc",
        action="store_true",
        help="keep the THC closure on even with --prognostic[-spherical]",
    )
    ap.add_argument(
        "--qflux",
        default=None,
        help="path to a frozen q-flux .npy -> FREE mode (q-flux + weak restoring) so "
        "SST/density can relax; needed for a realistic prognostic AMOC magnitude",
    )
    ap.add_argument(
        "--restore-tau-days",
        type=float,
        default=3650.0,
        help="weak anomaly-restoring timescale [days] in FREE mode (with --qflux)",
    )
    ap.add_argument(
        "--seasonal",
        action="store_true",
        help="P5: real seasonal cycle (insolation from the model day) instead of "
        "the legacy perpetual-equinox forcing",
    )
    ap.add_argument(
        "--orbit",
        choices=["pi", "6ka"],
        default="pi",
        help="P5: orbital configuration for --seasonal (pi=present-day, "
        "6ka=mid-Holocene)",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    end_day = args.days if args.days > 0 else int(round(args.years * DAYS_PER_YEAR))

    _mom_kw = {}
    if args.prognostic:
        _mom_kw = dict(
            prognostic_momentum=True, mom_drag=1.0 / (86400.0 * args.mom_drag_days)
        )
        if not args.keep_thc:
            _mom_kw["thc_k_vel"] = 0.0
    if args.prognostic_spherical:
        _mom_kw.update(prognostic_spherical=True, ah=args.ocean_ah)
        if not args.keep_thc:
            _mom_kw["thc_k_vel"] = 0.0
    if args.qflux is not None:
        # FREE mode: frozen q-flux + weak anomaly restoring so SST/density can RELAX
        # (the prognostic AMOC only reaches a realistic magnitude when the too-sharp WOA
        # density is allowed to adjust to the model's own equilibrium). Strong restoring
        # (default) pins the surface and keeps the overturning elevated.
        import numpy as _np

        _mom_kw.update(
            q_flux=jnp.asarray(_np.load(args.qflux)),
            restore_tau_days=args.restore_tau_days,
        )
        print(
            f"FREE mode: q-flux from {args.qflux} (mean "
            f"{float(jnp.mean(jnp.asarray(_np.load(args.qflux)))):.2f} W/m2), "
            f"restore_tau {args.restore_tau_days}d",
            flush=True,
        )
    if args.seasonal:
        _orbit = {"pi": orbital.ORBIT_PI, "6ka": orbital.ORBIT_6KA}[args.orbit]
        _mom_kw.update(seasonal=True, orbit=_orbit)
        print(
            f"SEASONAL cycle ON, orbit={args.orbit} "
            f"(obliquity {_orbit.obliquity_deg} deg, ecc {_orbit.eccentricity}, "
            f"perihelion {_orbit.long_perihelion_deg} deg)",
            flush=True,
        )
    model = DinoCoupledModel(ocean_ic="woa", interval=args.interval, **_mom_kw)
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

    print(
        f"Library dino control run: days {day}->{end_day} (interval {args.interval}d, "
        f"checkpoint every {args.ckpt_every_days}d)",
        flush=True,
    )

    # windowed q-flux accumulator: the time-mean restoring heat flux over each
    # checkpoint window IS the implied flux correction. Saved per checkpoint so the
    # converged window gives the q-flux for the FREE (forcing-responsive) run.
    qf_sum = np.zeros((OCEAN_GRID.nlat, OCEAN_GRID.nlon))
    qf_n = 0

    n_intervals = int(round((end_day - day) / args.interval))
    for it in range(1, n_intervals + 1):
        cstate = model.step_fast(cstate, co2_ppm=280.0)  # control: zero forcing
        day = int(round(cstate.day))
        if model.sst_target is not None:
            qf_sum += np.asarray(
                flux_correction.restoring_flux(
                    cstate.ocean.temp[0], model.sst_target, model.restore_tau_days
                )
            )
            qf_n += 1
        finite = bool(
            np.isfinite(np.asarray(cstate.ocean.temp)).all()
            and np.isfinite(np.asarray(cstate.atmos.vorticity)).all()
        )
        if (day % 30 == 0) or (it == n_intervals) or (not finite):
            ice_area = float(jnp.sum(jnp.where(omask, cstate.ice.concentration, 0.0)))
            amoc = float(
                compute_amoc(
                    cstate.ocean,
                    ocean_mask=omask,
                    ocean_mask_3d=model.ocean_mask_3d,
                )["upper_cell_26N"]
            )
            print(
                f"  day {day:6d} ({day / DAYS_PER_YEAR:6.2f} yr): "
                f"SST {sst_mean_C(cstate):5.2f}C  AMOC {amoc:5.1f}Sv  "
                f"ice-area {ice_area:8.0f}  "
                f"|curr|max {float(jnp.abs(cstate.ocean.u).max()):.3f}  finite {finite}",
                flush=True,
            )
        if not finite:
            save_state(cstate, _ckpt_base(args.outdir, day) + "_NAN")
            raise FloatingPointError(f"non-finite state at day {day}; dump written")
        if (day % args.ckpt_every_days == 0) or (it == n_intervals):
            save_state(cstate, _ckpt_base(args.outdir, day))
            msg = f"  [checkpoint] day {day} -> {_ckpt_base(args.outdir, day)}.npz"
            if qf_n > 0:
                np.save(_ckpt_base(args.outdir, day) + "_qflux.npy", qf_sum / qf_n)
                msg += f" (+_qflux.npy, mean of {qf_n})"
                qf_sum[:] = 0.0
                qf_n = 0
            print(msg, flush=True)

    print(f"done: ran to day {day} ({day / DAYS_PER_YEAR:.2f} yr).", flush=True)


if __name__ == "__main__":
    main_cli()
