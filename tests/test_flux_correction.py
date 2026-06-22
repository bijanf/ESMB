"""P2 tests: ocean SST flux-correction (strong restoring vs frozen q-flux + weak).

Pure-jnp, CI-runnable (no ETOPO / model build).
"""

import jax
import jax.numpy as jnp
import numpy as np

from chronos_esm.config import CP_WATER, OCEAN_DZ, RHO_WATER
from chronos_esm.ocean import flux_correction as fc


def test_restoring_lambda_value():
    # lambda = rho*cp*dz_surf/tau; ~50 W/m2/K for tau=30d, dz_surf~ tens of m.
    lam = fc.restoring_lambda(30.0)
    expect = RHO_WATER * CP_WATER * float(OCEAN_DZ[0]) / (30.0 * 86400.0)
    np.testing.assert_allclose(lam, expect, rtol=1e-6)
    assert lam > 0


def test_restoring_flux_sign_and_zero():
    sst = jnp.array([[290.0, 295.0]])
    tgt = jnp.array([[291.0, 295.0]])  # col0 colder than target, col1 on target
    f = fc.restoring_flux(sst, tgt, 30.0)
    assert float(f[0, 0]) > 0.0  # heat INTO ocean where it is too cold
    assert float(f[0, 1]) == 0.0  # no flux when at target


def test_qflux_mode_delivers_frozen_flux():
    # FREE mode: when SST == target, the weak restoring term is 0, so the correction
    # is exactly the prescribed q-flux (the frozen mean correction is delivered).
    sst = jnp.array([[288.0]])
    tgt = jnp.array([[288.0]])
    qbar = jnp.array([[42.0]])
    corr = fc.heat_correction(sst, tgt, restore_tau_days=3650.0, q_flux=qbar)
    np.testing.assert_allclose(float(corr[0, 0]), 42.0, atol=1e-5)


def test_qflux_weak_restoring_is_small_vs_strong():
    # a 1 K anomaly drives a much weaker correction at long tau (free) than short (control).
    sst, tgt = jnp.array([[289.0]]), jnp.array([[290.0]])
    strong = float(fc.restoring_flux(sst, tgt, 30.0)[0, 0])
    weak = float(fc.heat_correction(sst, tgt, 3650.0, q_flux=jnp.zeros((1, 1)))[0, 0])
    assert 0 < weak < 0.05 * strong  # ~tau ratio 30/3650


def test_heat_correction_differentiable_in_sst():
    sst = jnp.array([[289.0]])
    tgt = jnp.array([[290.0]])
    g = float(jax.grad(lambda s: fc.heat_correction(s, tgt, 30.0).sum())(sst)[0, 0])
    np.testing.assert_allclose(
        g, -fc.restoring_lambda(30.0), rtol=1e-5
    )  # d/dSST = -lambda
