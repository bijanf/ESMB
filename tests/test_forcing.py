"""P2 tests: single-channel CO2 radiative forcing into the ocean heat budget.

These are pure-jnp (no ETOPO / model build), so they run in CI -- giving the
forcing-response path real coverage even where the heavy integration tests skip.
"""
import jax
import jax.numpy as jnp
import numpy as np

from chronos_esm.coupler.dino_coupling import co2_forcing_wm2, ocean_fluxes_jax, LAT_LIN


def test_co2_forcing_myhre_values():
    assert float(co2_forcing_wm2(280.0)) == 0.0                       # no forcing at C0
    np.testing.assert_allclose(float(co2_forcing_wm2(560.0)), 3.708, atol=1e-2)  # 2xCO2
    # log-linear: 4xCO2 forcing is twice 2xCO2 forcing
    np.testing.assert_allclose(float(co2_forcing_wm2(1120.0)),
                               2.0 * float(co2_forcing_wm2(560.0)), rtol=1e-6)


def _synthetic():
    nlat, nlon = LAT_LIN.shape[0], 8
    rng = np.random.default_rng(0)
    f = lambda s=1.0, b=0.0: jnp.asarray((b + s * rng.standard_normal((nlat, nlon))).astype(np.float32))
    return (f(5, 290), f(5), f(5), f(5, 288), jnp.abs(f(1e-3)), jnp.abs(f(1e-4)))


def test_co2_forcing_adds_heat_to_budget():
    sst, u, v, ta, qa, pr = _synthetic()
    # free budget (no flux-correction, no mean-removal) so the uniform forcing survives.
    base = ocean_fluxes_jax(sst, u, v, ta, qa, pr, balance_heat=False)[0]
    forced = ocean_fluxes_jax(sst, u, v, ta, qa, pr, balance_heat=False, co2_ppm=560.0)[0]
    diff = float(jnp.mean(forced - base))
    np.testing.assert_allclose(diff, 3.708, atol=1e-2)   # ~ +3.71 W/m2 everywhere


def test_co2_forcing_differentiable():
    sst, u, v, ta, qa, pr = _synthetic()

    def mean_net_heat(co2):
        nh = ocean_fluxes_jax(sst, u, v, ta, qa, pr, balance_heat=False, co2_ppm=co2)[0]
        return jnp.mean(nh)

    g = float(jax.grad(mean_net_heat)(560.0))
    assert np.isfinite(g) and g > 0.0, f"d(net_heat)/d(CO2) should be finite>0, got {g}"
    np.testing.assert_allclose(g, 5.35 / 560.0, rtol=5e-3)   # d/dC [5.35 ln(C/C0)] = 5.35/C (float32)
