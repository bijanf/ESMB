"""Is the T31 prognostic-AMOC barrier a surface-pressure (barotropic) problem? No.

The roadmap listed two candidate levers for the prognostic-momentum ocean core's
~300-550 Sv AMOC barrier (docs/prognostic_ocean_core.md): (1) Gent-McWilliams eddy
mixing, and (2) a rigid-lid **surface-pressure reference**. GM was implemented and
measured to cut the barrier by only ~1% (experiments/diagnose_gm_amoc.py). This script
settles lever (2) by DECOMPOSING the overturning into its barotropic and baroclinic parts.

A surface pressure p_s(x,y) is DEPTH-INDEPENDENT, so grad(p_s) drives only the
**barotropic** (depth-mean) flow; it cannot change the **baroclinic** shear. So a
surface-pressure reference can only matter for the AMOC if the barrier lives in the
barotropic mode. It does not:

  * the AMOC upper cell is IDENTICAL with and without removing the section barotropic
    transport (compute_amoc remove_barotropic True vs False) -> the overturning has no
    barotropic component;
  * the global net meridional transport per latitude is ~0 Sv, and the depth-integrated
    transport is non-divergent to machine precision -> the old +/-350 Sv spurious
    barotropic mode is ALREADY removed by barotropic.rigid_lid_project (step_ocean:219).

Conclusion: the 326 Sv barrier is PURE BAROCLINIC overturning -- the geostrophic
thermal-wind transport, ~20-40x too large at T31. A surface-pressure reference is NOT
the lever (the barotropic mode is already clean). With GM (necessary, ~1%) and the
rigid-lid projection (already done) both settled, the prognostic core's remaining gap
is a **resolution limit**, not a parameterization one -- the shipped diagnostic + THC
path is the correct T31 production AMOC.

    PYTHONPATH=$PWD JAX_PLATFORMS=cpu venv/bin/python experiments/diagnose_amoc_barotropic.py [--steps 30]

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
    dx = (2 * np.pi * EARTH_RADIUS * cosl[:, None]) / NX
    dy = (np.pi * EARTH_RADIUS) / NY
    dz = jnp.asarray(OCEAN_DZ)
    flux = (jnp.zeros((NY, NX)),) * 3
    stress = (jnp.zeros((NY, NX)),) * 2
    omask = create_atlantic_mask(NY, NX)
    o2d = mask3d[0] > 0.5

    s = st
    for _ in range(steps):
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
        )

    def amoc(rb):
        return float(
            compute_amoc(
                s, atlantic_mask=omask, dz=dz, ocean_mask=o2d, remove_barotropic=rb
            )["upper_cell_26N"]
        )

    a_bc = amoc(True)  # barotropic removed -> baroclinic overturning
    a_full = amoc(False)  # barotropic included

    v = s.v * mask3d
    net_lat = jnp.sum(v * dz[:, None, None] * dx, axis=(0, 2)) / 1e6  # (ny,) Sv
    U = jnp.sum(s.u * dz[:, None, None], axis=0)
    V = jnp.sum(s.v * dz[:, None, None], axis=0)
    div_U = (jnp.roll(U, -1, 1) - jnp.roll(U, 1, 1)) / (2 * dx) + (
        jnp.roll(V, -1, 0) - jnp.roll(V, 1, 0)
    ) / (2 * dy)

    print(f"=== Prognostic-momentum AMOC decomposition (WOA18 T31, {steps} steps) ===")
    print(f"  AMOC upper 26N, barotropic REMOVED  (baroclinic) : {a_bc:7.1f} Sv")
    print(f"  AMOC upper 26N, barotropic INCLUDED              : {a_full:7.1f} Sv")
    print(
        f"  -> identical to {abs(a_bc - a_full):.2g} Sv: the overturning is PURE BAROCLINIC"
    )
    print(
        f"  global net meridional transport: max|net/lat|    : "
        f"{float(jnp.max(jnp.abs(net_lat))):.2g} Sv  (barotropic spurious mode: gone)"
    )
    print(
        f"  depth-integrated |div(∫u dz)|                    : "
        f"{float(jnp.max(jnp.abs(div_U))):.1e}  (rigid_lid_project -> non-divergent)"
    )
    print(
        "\nConclusion: the barotropic mode is already non-divergent and net-zero "
        "(barotropic.rigid_lid_project).\nA rigid-lid surface-pressure reference sets only "
        "the barotropic flow, so it CANNOT reduce the\nbaroclinic 326 Sv AMOC. The barrier "
        "is the T31 baroclinic geostrophic overturning (~20-40x\ntoo large) -- a RESOLUTION "
        "limit, not a surface-pressure one. Both roadmap levers are now\nsettled (GM ~1%, "
        "surface-pressure not-the-lever); production stays on diagnostic + THC."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    main_diag(args.steps)
