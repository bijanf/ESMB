"""Regression test for the AMOC metric sign fix (P3 / S0).

A physically-correct overturning cell (northward NADW upper limb, equatorward deep
return) must score POSITIVE at 26.5N per the RAPID convention. A sign error in
compute_amoc (Psi = -cumsum, upper = max(Psi)) previously scored a clean ~15 Sv cell
as ~0 Sv -- a large part of the "AMOC ~ 0 / incoherent" story. This pins the fix.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chronos_esm.config import EARTH_RADIUS, OCEAN_DZ, OCEAN_GRID
from chronos_esm.ocean.diagnostics import compute_amoc, create_atlantic_mask
from chronos_esm.ocean.veros_driver import OceanState


def _cell_state(scale_sign=+1.0, target_sv=15.0):
    """Synthetic ocean with a clean Atlantic overturning cell: northward (sign +) in
    the upper ~1100 m, equatorward below, depth-integral zero (pure overturning)."""
    ny, nx, nz = OCEAN_GRID.nlat, OCEAN_GRID.nlon, len(OCEAN_DZ)
    dz = np.asarray(OCEAN_DZ)
    lat = np.linspace(-90, 90, ny)
    dx = 2 * np.pi * EARTH_RADIUS * np.cos(np.deg2rad(lat)) / nx
    atl = np.asarray(create_atlantic_mask(ny, nx))
    j26 = int(np.argmin(np.abs(lat - 26.5)))

    up = (np.cumsum(dz) <= 1100.0).astype(float)
    prof = up - (up * dz).sum() / dz.sum()  # depth-integral zero
    width = atl[j26].sum() * dx[j26]
    scale = (
        scale_sign * target_sv * 1e6 / ((np.where(up > 0, prof, 0) * dz).sum() * width)
    )
    v = np.stack([scale * prof[k] * atl for k in range(nz)], axis=0)
    z = jnp.zeros((nz, ny, nx))
    return OceanState(
        u=z,
        v=jnp.asarray(v),
        w=z,
        temp=jnp.full((nz, ny, nx), 283.0),
        salt=jnp.full((nz, ny, nx), 35.0),
        psi=jnp.zeros((ny, nx)),
        rho=z,
        dic=z,
    )


def test_amoc_metric_scores_clean_cell_positive():
    """A clean ~15 Sv northward-upper Atlantic cell scores ~+15 Sv (not ~0)."""
    r = compute_amoc(_cell_state(+1.0, 15.0))
    assert float(r["upper_cell_26N"]) == pytest.approx(
        15.0, abs=0.5
    ), f"upper_cell should be ~+15 Sv, got {float(r['upper_cell_26N']):+.3f}"


def test_amoc_metric_sign_distinguishes_anti_amoc():
    """The metric must distinguish a real AMOC from a reversed circulation: a real
    cell scores a large POSITIVE upper cell; a reversed cell scores ~0 upper and a
    strong NEGATIVE lower cell (so the two are never confused)."""
    real = compute_amoc(_cell_state(+1.0, 15.0))
    anti = compute_amoc(_cell_state(-1.0, 15.0))
    assert float(real["upper_cell_26N"]) > 10.0  # real AMOC: strong + upper
    assert float(anti["upper_cell_26N"]) < 1.0  # reversed: NOT a strong + upper
    assert float(anti["lower_cell_26N"]) < -10.0  # reversed shows as -lower
