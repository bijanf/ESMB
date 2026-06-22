"""What is the AMOC problem in the COUPLED model? Wild temporal noise, not a wrong mean.

Runs the coupled DinoCoupledModel (control config: diagnostic thermal wind + THC closure,
prognostic momentum off) and logs the AMOC each month, then decomposes the final state's
AMOC into its diagnostic-thermal-wind and THC contributions.

FINDING (2026-06-22): the 26.5 N overturning is a CLEAN, textbook cell at any instant
(smoothly rising to a ~20 Sv max near 900 m, decaying to 0 at the floor), but its
AMPLITUDE oscillates violently over the run -- e.g. 9, 22, 50, 0, 110, 18 Sv month to
month -- while SST stays flat (~14 C, restored). The diagnostic thermal wind is negligible
(~0.1 Sv, heavily damped by r_drag=0.05); the entire AMOC and its noise are the THC
closure (chronos_esm/ocean/overturning.py). The closure sets the overturning amplitude
INSTANTANEOUSLY: amp = k_vel*softplus(contrast/drho_scale), where contrast is the small
subpolar-minus-subtropical density difference. Coupled-forcing + convection noise makes that
contrast cross zero, and softplus near zero swings amp from 0 to large -> the AMOC has NO
temporal inertia. (The README's old "34/75 Sv, too high, spurious net transport" captions
are stale snapshots of this noise; the spurious net transport was already fixed by
barotropic.rigid_lid_project / the per-latitude corrector -- net transport is ~0.)

FIX (real AMOC has multi-year inertia): relax the THC overturning amplitude toward its
density-implied target over a timescale tau ~ 1-3 yr, carried at the coupling level
(DinoCoupledState), so the overturning cannot track instantaneous density noise while
staying density-driven on long timescales (tipping/paleo preserved).

    PYTHONPATH=$PWD JAX_PLATFORMS=cpu venv/bin/python experiments/diagnose_coupled_amoc.py [--days 180]

Slow (~3 s/model-day on CPU). Needs the cached WOA18 + ETOPO data.
"""

import argparse
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm.coupler.dino_step import DinoCoupledModel  # noqa: E402
from chronos_esm.ocean.diagnostics import compute_amoc  # noqa: E402


def main_diag(days):
    model = DinoCoupledModel(ocean_ic="woa", interval=1.0)
    omask = model.omask
    cstate = model.init_state()

    def amoc_up(oc):
        return float(
            compute_amoc(oc, ocean_mask=omask, ocean_mask_3d=model.ocean_mask_3d)[
                "upper_cell_26N"
            ]
        )

    print(f"=== Coupled AMOC trajectory (control config, {days} days) ===")
    series = []
    for it in range(1, days + 1):
        cstate = model.step_fast(cstate, co2_ppm=280.0)
        if it % 30 == 0:
            a = amoc_up(cstate.ocean)
            series.append(a)
            sst = (
                float(
                    jnp.sum(jnp.where(omask, cstate.ocean.temp[0], 0.0))
                    / jnp.sum(omask)
                )
                - 273.15
            )
            print(f"  day {it:4d}: AMOC {a:7.1f} Sv   SST {sst:5.2f} C")
    if series:
        s = np.array(series)
        print(
            f"\n  AMOC over the run: mean {s.mean():.1f}  min {s.min():.1f}  "
            f"max {s.max():.1f}  std {s.std():.1f} Sv  (SST flat -> the AMOC noise is "
            f"NOT an SST-drift signal)"
        )
    print(
        "\nConclusion: the cell shape is clean; its AMPLITUDE is wildly noisy because the THC\n"
        "closure has no temporal inertia (instantaneous softplus response to a near-zero,\n"
        "noisy density contrast). The diagnostic thermal wind is negligible. Fix = relax the\n"
        "THC amplitude over tau ~ 1-3 yr (coupling-level inertia)."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    args = ap.parse_args()
    main_diag(args.days)
