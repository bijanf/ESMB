"""P0 tests for the differentiable dino<->ocean coupler (chronos_esm/coupler/dino_coupling).

Verifies the jnp-native re-implementations are (a) numerically equivalent to the
numpy harness helpers they replace, and (b) actually differentiable end-to-end
through one ocean coupling interval — the foundation every downstream workstream
(forcing, tipping, paleo-DA, carbon) depends on.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pooch
import pytest

# init_model() + DinoAtmosphere(orography=True) pull the ~900 MB ETOPO bathymetry;
# skip if it isn't cached (e.g. in CI). Stage with experiments/prefetch_data.py.
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(pooch.os_cache("chronos_esm"), "etopo1.nc")),
    reason="ETOPO1 (~900 MB) not cached; run experiments/prefetch_data.py to enable",
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "experiments"))


@pytest.fixture(scope="module")
def atm():
    from chronos_esm.atmos.dino_atmos import DinoAtmosphere
    return DinoAtmosphere()


@pytest.fixture(scope="module")
def dino_state(atm):
    # a non-trivial state: 2 days of evolution under a simple SST gradient so winds,
    # humidity and precip are all populated (exercises every diagnostic field).
    sst_g = jnp.asarray(300.0 - 35.0 * atm.sin_lat ** 2)   # (nlon, nlat) K
    return atm.step(atm.initial_state(sst_g), sst_g, n_days=2)


def test_diagnostics_jax_matches_numpy(atm, dino_state):
    from chronos_esm.coupler.dino_coupling import dino_diagnostics_jax
    ref = atm.diagnostics(dino_state)                 # numpy/pint reference
    got = dino_diagnostics_jax(atm, dino_state)        # jnp port
    for k in ("u_sfc", "v_sfc", "t_sfc", "q_sfc", "surface_pressure", "mslp", "precip"):
        a = np.asarray(ref[k])
        b = np.asarray(got[k])
        assert a.shape == b.shape, f"{k} shape {a.shape} vs {b.shape}"
        # float32 spectral transforms -> match to a tight relative tolerance.
        np.testing.assert_allclose(b, a, rtol=2e-4, atol=1e-3,
                                   err_msg=f"jnp diagnostics[{k}] != numpy")


def test_regridders_match_numpy(atm):
    import run_dino_coupled as rdc
    from chronos_esm.coupler.dino_coupling import make_regridders_jax, LAT_LIN

    np_l2g, np_g2l = rdc.make_regridders(atm.lat_deg)
    jx_l2g, jx_g2l = make_regridders_jax(atm.lat_deg)

    nlat_lin = LAT_LIN.shape[0]
    rng = np.random.default_rng(0)
    f_lin = rng.standard_normal((nlat_lin, atm.nlon)).astype(np.float32)
    f_g = rng.standard_normal((atm.nlon, atm.nlat)).astype(np.float32)

    np.testing.assert_allclose(np.asarray(jx_l2g(jnp.asarray(f_lin))),
                               np_l2g(f_lin), rtol=1e-5, atol=1e-4)
    np.testing.assert_allclose(np.asarray(jx_g2l(jnp.asarray(f_g))),
                               np_g2l(f_g), rtol=1e-5, atol=1e-4)


def test_ocean_fluxes_match_numpy(atm):
    import run_dino_coupled as rdc
    from chronos_esm.coupler.dino_coupling import ocean_fluxes_jax, LAT_LIN

    nlat, nlon = LAT_LIN.shape[0], atm.nlon
    rng = np.random.default_rng(1)
    sst = (290.0 + 5.0 * rng.standard_normal((nlat, nlon))).astype(np.float32)
    u = rng.standard_normal((nlat, nlon)).astype(np.float32) * 5
    v = rng.standard_normal((nlat, nlon)).astype(np.float32) * 5
    ta = (288.0 + 5.0 * rng.standard_normal((nlat, nlon))).astype(np.float32)
    qa = np.abs(rng.standard_normal((nlat, nlon)).astype(np.float32)) * 1e-3
    pr = np.abs(rng.standard_normal((nlat, nlon)).astype(np.float32)) * 1e-4
    omask = (rng.standard_normal((nlat, nlon)) > 0.0)
    sst_t = sst + 0.5

    # both code paths: SST flux-correction (sst_target) and ocean-mean balance.
    for kwargs in (dict(sst_target=sst_t, ocean_mask=omask),
                   dict(balance_heat=True, ocean_mask=omask)):
        ref = rdc.ocean_fluxes(sst, u, v, ta, qa, pr, **kwargs)
        got = ocean_fluxes_jax(jnp.asarray(sst), jnp.asarray(u), jnp.asarray(v),
                               jnp.asarray(ta), jnp.asarray(qa), jnp.asarray(pr),
                               **{k: (jnp.asarray(val) if isinstance(val, np.ndarray) else val)
                                  for k, val in kwargs.items()})
        for name, a, b in zip(("net_heat", "fw", "tau_x", "tau_y"), ref, got):
            np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-4, atol=1e-3,
                                       err_msg=f"ocean_fluxes[{name}] mismatch ({kwargs.keys()})")


@pytest.fixture(scope="module")
def ocean_interval_f():
    """A short, checkpointed ocean interval as one differentiable function of a
    subpolar-Atlantic surface-salinity perturbation -> amoc_strength [Sv].
    Module-scoped so init_model() (heavy: ~900 MB ETOPO + WOA) is built ONCE and
    shared by both gradient tests, keeping the whole-file memory footprint bounded."""
    return _ocean_interval_fn(n_sub=6)


def _ocean_interval_fn(n_sub=6):
    from chronos_esm import main
    from chronos_esm.config import OCEAN_DZ, OCEAN_GRID, EARTH_RADIUS
    from chronos_esm.ocean import veros_driver
    from chronos_esm.ocean.diagnostics import create_atlantic_mask
    from chronos_esm.coupler.dino_coupling import amoc_strength

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

    zero = (jnp.zeros((OCEAN_GRID.nlat, OCEAN_GRID.nlon)),) * 2
    fluxes = (zero[0], zero[0], zero[0])
    wind = (zero[0], zero[1])
    surf_ocean_mask = jnp.asarray(np.asarray(surface_mask).astype(bool))

    @jax.checkpoint
    def body(oc, _):
        return veros_driver.step_ocean(
            oc, surface_fluxes=fluxes, wind_stress=wind, dx=dx, dy=dy, dz=dz,
            nz=nz, mask=surface_mask, ocean_mask_3d=ocean_mask_3d), None

    def f(eps):
        # subpolar freshening: add eps psu to surface salinity in the subpolar Atlantic
        salt = ocean.salt.at[0].add(eps * subpolar)
        oc = ocean._replace(salt=salt)
        oc, _ = jax.lax.scan(body, oc, None, length=n_sub)
        return amoc_strength(oc, ocean_mask=surf_ocean_mask)

    return f


def test_grad_through_ocean_interval(ocean_interval_f):
    """value_and_grad through a (checkpointed) ocean interval returns finite,
    non-NaN gradients on CPU — i.e. the ocean half of the coupled model is
    end-to-end differentiable."""
    val, grad = jax.value_and_grad(ocean_interval_f)(0.0)
    assert np.isfinite(float(val)), f"AMOC value non-finite: {val}"
    assert np.isfinite(float(grad)), f"gradient non-finite (graph broken): {grad}"


def test_amoc_density_responsive(ocean_interval_f):
    """P3/S1: the thermohaline overturning closure makes the AMOC DENSITY-RESPONSIVE.
    d(AMOC@26.5N)/d(subpolar salinity) is now nonzero and sign-correct (saltier
    subpolar -> denser -> stronger AMOC), AD matching finite-difference. (Pre-S1 it
    was ~0: the diagnostic thermal wind carried no overturning to 26.5N -- this test
    was the regression baseline that asserted abs(grad) < 1e-3.)"""
    grad = float(jax.grad(ocean_interval_f)(0.0))
    eps = 0.1
    fd = (float(ocean_interval_f(eps)) - float(ocean_interval_f(-eps))) / (2 * eps)
    assert grad > 1e-3, f"AMOC should be density-responsive (grad>0), got {grad:.3e}"
    assert abs(grad - fd) <= 0.05 * abs(fd) + 1e-6, f"AD {grad:.3e} != FD {fd:.3e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
