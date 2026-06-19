"""P0 regression baseline: d(AMOC@26.5N)/d(subpolar-Atlantic density) on the CURRENT
(diagnostic-velocity) ocean.

The ocean's velocities are DIAGNOSTIC -- rebuilt every step from a wind-driven
Stommel barotropic streamfunction plus a thermal-wind baroclinic anomaly whose net
per-latitude transport is then removed (veros_driver.py), and compute_amoc removes
the barotropic mode again. The consequence, which this script measures and RECORDS,
is that density has essentially NO pathway to net overturning:

    AMOC@26.5N ~= 0 Sv   and   d(AMOC)/d(subpolar salinity) ~= 0

This is the verified blocker behind "AMOC tipping is impossible today" (no hosing or
buoyancy forcing can move an overturning density cannot drive). It is the regression
baseline that the P3 prognostic-momentum ocean core MUST break: after P3 this
gradient should be a non-zero, sign-correct sensitivity.

    python experiments/p0_amoc_gradient_baseline.py [--substeps N]

Reports the AMOC value, the reverse-mode (jax.grad) sensitivity, and a central
finite-difference cross-check.
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm import main  # noqa: E402
from chronos_esm.config import OCEAN_DZ, OCEAN_GRID, EARTH_RADIUS  # noqa: E402
from chronos_esm.ocean import veros_driver  # noqa: E402
from chronos_esm.ocean.diagnostics import create_atlantic_mask, compute_amoc  # noqa: E402
from chronos_esm.coupler.dino_coupling import amoc_strength  # noqa: E402


def build_amoc_of_eps(n_sub):
    """Return f(eps) = AMOC@26.5N [Sv] after n_sub ocean substeps, where eps [psu]
    freshens the subpolar (45-65N) Atlantic surface salinity."""
    state = main.init_model(ocean_ic="woa")
    ocean = state.ocean
    nz = ocean.u.shape[0]
    ocean_mask_3d, surface_mask = main.ocean_masks(nz=nz)

    lat = np.linspace(-90, 90, OCEAN_GRID.nlat)
    dy = (np.pi * EARTH_RADIUS) / OCEAN_GRID.nlat
    cos_lat = np.maximum(np.cos(np.deg2rad(lat)), 0.05)
    dx = jnp.asarray((2 * np.pi * EARTH_RADIUS * cos_lat[:, None]) / OCEAN_GRID.nlon)
    dz = jnp.asarray(OCEAN_DZ)

    atl = np.asarray(create_atlantic_mask(OCEAN_GRID.nlat, OCEAN_GRID.nlon))
    subpolar = jnp.asarray(atl & (lat[:, None] >= 45.0) & (lat[:, None] <= 65.0))
    surf_ocean_mask = jnp.asarray(np.asarray(surface_mask).astype(bool))

    z2 = jnp.zeros((OCEAN_GRID.nlat, OCEAN_GRID.nlon))
    fluxes = (z2, z2, z2)
    wind = (z2, z2)

    @jax.checkpoint
    def body(oc, _):
        return veros_driver.step_ocean(
            oc, surface_fluxes=fluxes, wind_stress=wind, dx=dx, dy=dy, dz=dz,
            nz=nz, mask=surface_mask, ocean_mask_3d=ocean_mask_3d), None

    def f(eps):
        salt = ocean.salt.at[0].add(eps * subpolar)
        oc = ocean._replace(salt=salt)
        oc, _ = jax.lax.scan(body, oc, None, length=n_sub)
        return amoc_strength(oc, ocean_mask=surf_ocean_mask)

    return f, (ocean, surf_ocean_mask)


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--substeps", type=int, default=24,
                    help="ocean substeps (900s each) before diagnosing AMOC")
    ap.add_argument("--eps", type=float, default=0.5, help="FD perturbation [psu]")
    args = ap.parse_args()

    f, (ocean0, surf_mask) = build_amoc_of_eps(args.substeps)

    amoc0 = float(amoc_strength(ocean0, ocean_mask=surf_mask))
    val, grad = jax.value_and_grad(f)(0.0)
    fd = (float(f(args.eps)) - float(f(-args.eps))) / (2.0 * args.eps)

    print("=" * 64)
    print("P0 BASELINE  --  d(AMOC@26.5N)/d(subpolar-Atlantic salinity)")
    print("=" * 64)
    print(f"  AMOC@26.5N (unperturbed, t=0)        : {amoc0:+.4f} Sv")
    print(f"  AMOC@26.5N (after {args.substeps:3d} substeps)       : {float(val):+.4f} Sv")
    print(f"  d(AMOC)/d(salt)  reverse-mode (grad) : {float(grad):+.4e} Sv/psu")
    print(f"  d(AMOC)/d(salt)  finite-difference   : {fd:+.4e} Sv/psu")
    print("-" * 64)
    near_zero = abs(float(grad)) < 1e-3
    print(f"  VERDICT: density {'has NO' if near_zero else 'HAS A'} pathway to net "
          f"overturning  (|grad| {'<' if near_zero else '>='} 1e-3)")
    if near_zero:
        print("  -> matches the documented blocker; P3 (prognostic momentum) must")
        print("     break this to a non-zero, sign-correct d(AMOC)/d(density).")
    else:
        print("  -> a real density->overturning sensitivity exists; if P3 has landed,")
        print("     update tests/test_dino_coupling.py::test_amoc_density_gradient_baseline.")
    print("=" * 64)


if __name__ == "__main__":
    main_cli()
