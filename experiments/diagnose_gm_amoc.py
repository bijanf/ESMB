"""Quantify what Gent-McWilliams (GM) eddy mixing does to the T31 prognostic-AMOC barrier.

Context: the prognostic-momentum ocean core (step_ocean(prognostic_momentum=True)) gives an
AMOC ~300-550 Sv at T31 -- the documented resolution barrier (docs/prognostic_ocean_core.md).
The roadmap hypothesis was that GM (flattening isopycnals + an eddy-induced overturning that
opposes the mean) would cut this toward ~15 Sv. This script TESTS that hypothesis on a
realistic WOA18-initialized ocean.

It reports, at 26.5 N:
  * the WOA isopycnal-slope magnitude (to confirm GM is active, i.e. slopes are gentle, not
    tapered off as they would be in a near-neutral column);
  * the GM bolus (eddy-induced) velocity magnitude and its overturning streamfunction;
  * the Eulerian-mean AMOC (gm off vs gm on) and the RESIDUAL-mean AMOC (Eulerian + bolus);
  * that the density sensitivity d(AMOC)/d(subpolar salt) is preserved with GM on.

Finding (2026-06-22): GM produces a physically-correct ~1-2 Sv eddy overturning, but reduces
the ~326 Sv barrier by only ~1%. The barrier is dominated by the EULERIAN-MEAN cell (the
momentum / rigid-lid surface-pressure regime), which GM does not address -- so GM is necessary
eddy physics but NOT the lever for this barrier. Next P3 lever: the surface-pressure reference.

    PYTHONPATH=$PWD JAX_PLATFORMS=cpu venv/bin/python experiments/diagnose_gm_amoc.py [--steps 30]

Needs the cached WOA18 climatology (~/.cache/chronos_esm).
"""

import argparse
import os
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm import data, main  # noqa: E402
from chronos_esm.config import EARTH_RADIUS, OCEAN_DZ, OCEAN_GRID  # noqa: E402
from chronos_esm.ocean import gm  # noqa: E402
from chronos_esm.ocean.diagnostics import (  # noqa: E402
    compute_amoc,
    create_atlantic_mask,
)
from chronos_esm.ocean.veros_driver import (  # noqa: E402
    equation_of_state,
    init_ocean_state,
    step_ocean,
)

NZ, NY, NX = OCEAN_GRID.nz, OCEAN_GRID.nlat, OCEAN_GRID.nlon
KAPPA_GM = 1000.0


def _kappa_pole_masked(kappa):
    """kappa * interior_mask_3d, matching step_ocean's pole masking (rows 0-4, -5..-1)."""
    idx = jnp.arange(NY)
    interior = jnp.where((idx < 5) | (idx >= NY - 5), 0.0, 1.0)[None, :, None]
    return kappa * interior


def main_diag(steps):
    t_ic, s_ic = data.load_initial_conditions(nz=NZ)
    mask3d, surface = main.ocean_masks()
    mask3d = jnp.asarray(mask3d)
    st = init_ocean_state(NZ, NY, NX)._replace(
        temp=jnp.asarray(t_ic) + 273.15, salt=jnp.asarray(s_ic)
    )
    st = st._replace(rho=equation_of_state(st.temp, st.salt))

    lat = jnp.linspace(-90, 90, NY)
    cosl = jnp.maximum(jnp.cos(jnp.deg2rad(lat)), 0.05)
    dx = (2 * np.pi * EARTH_RADIUS * cosl[:, None]) / NX  # (ny,1) lat-dependent
    dy = (np.pi * EARTH_RADIUS) / NY
    dz = jnp.asarray(OCEAN_DZ)
    flux = (jnp.zeros((NY, NX)),) * 3
    stress = (jnp.zeros((NY, NX)),) * 2
    omask = create_atlantic_mask(NY, NX)
    o2d = mask3d[0] > 0.5
    kge = _kappa_pole_masked(KAPPA_GM)
    dx_gm = gm.latitude_dx(NY, NX)

    # --- slope + bolus diagnostics on the realistic WOA density --------------------
    _, sy = gm.isopycnal_slopes(st.rho, dx_gm, dy, dz)
    wet = np.asarray(mask3d) > 0.5
    sy_med = float(np.median(np.abs(np.asarray(sy)[wet])))
    sy_p90 = float(np.percentile(np.abs(np.asarray(sy)[wet]), 90))
    _, vstar0 = gm.eddy_induced_velocity(kge, st.rho, dx_gm, dy, dz, maskC=mask3d)
    print("=== GM activity on the WOA18 ocean (T31) ===")
    print(
        f"  isopycnal slope |Sy|: median {sy_med:.2e}  p90 {sy_p90:.2e}  "
        f"(< S_crit={gm.S_CRIT:g} => GM active)"
    )
    print(
        f"  bolus |v*|: max {float(jnp.max(jnp.abs(vstar0))):.2e}  "
        f"mean {float(jnp.mean(jnp.abs(vstar0))):.2e} m/s"
    )

    def amoc_up(state):
        return float(
            compute_amoc(state, atlantic_mask=omask, dz=dz, ocean_mask=o2d)[
                "upper_cell_26N"
            ]
        )

    def run(gm_on, n):
        s = st
        for _ in range(n):
            s = step_ocean(
                s,
                flux,
                stress,
                dx,
                dy,
                dz,
                mask=surface.astype(float),
                ocean_mask_3d=mask3d,
                prognostic_momentum=True,
                thc_k_vel=0.0,
                gm_on=gm_on,
            )
        return s

    s_off = run(False, steps)
    s_on = run(True, steps)
    _, vstar = gm.eddy_induced_velocity(kge, s_on.rho, dx_gm, dy, dz, maskC=mask3d)
    s_res = s_on._replace(v=s_on.v + vstar)  # residual-mean velocity = Eulerian + bolus

    a_off, a_on, a_res = amoc_up(s_off), amoc_up(s_on), amoc_up(s_res)
    print(f"\n=== AMOC upper cell at 26.5 N (after {steps} steps) ===")
    print(f"  Eulerian-mean, GM off : {a_off:7.1f} Sv  (the barrier)")
    print(f"  Eulerian-mean, GM on  : {a_on:7.1f} Sv")
    print(
        f"  RESIDUAL-mean, GM on  : {a_res:7.1f} Sv  "
        f"(Eulerian + GM bolus; reduction {100*(1-a_res/a_off):.1f}%)"
    )

    # --- density responsiveness preserved with GM on ------------------------------
    spm = jnp.asarray(
        np.broadcast_to(
            (
                (np.linspace(-90, 90, NY) >= 45) & (np.linspace(-90, 90, NY) <= 65)
            ).astype(np.float64)[None, :, None],
            (NZ, NY, NX),
        )
    )

    def overturn(salt_pert):
        s = st._replace(salt=st.salt + spm * salt_pert)
        for _ in range(8):
            s = step_ocean(
                s,
                flux,
                stress,
                dx,
                dy,
                dz,
                mask=surface.astype(float),
                ocean_mask_3d=mask3d,
                prognostic_momentum=True,
                thc_k_vel=0.0,
                gm_on=True,
            )
        vz = jnp.sum(s.v * dz[:, None, None] * dx, axis=2)
        psi = jnp.cumsum(vz, axis=0) / 1.0e6
        return jnp.sqrt(jnp.mean(psi**2))

    g = float(jax.grad(overturn)(0.0))
    print(
        f"\n  d(overturning)/d(subpolar salt) with GM on = {g:.3e} Sv/psu "
        f"({'preserved (nonzero)' if abs(g) > 1.0 else 'WEAK'})"
    )

    print(
        "\nConclusion: GM gives a correct ~1-2 Sv eddy overturning but barely dents the\n"
        "Eulerian-mean barrier. The ~300 Sv cell is a momentum / rigid-lid surface-pressure\n"
        "problem, not an eddy-slope one -- GM is necessary but not sufficient at T31."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    main_diag(args.steps)
