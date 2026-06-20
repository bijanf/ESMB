"""P2 forcing-path integration tests: the single-channel CO2 forcing survives the
SST flux-correction in BOTH the strongly-restored CONTROL config and the frozen
q-flux + weak-restoring FREE config.

These are pure-jnp (no model build, no ETOPO) and run in well under 10 s. They prove
end-to-end that ``ocean_fluxes_jax`` adds the Myhre CO2 forcing to the ocean surface
net heat budget, and -- crucially -- that the flux-correction (which does NOT depend
on co2_ppm) CANCELS exactly in the forced-minus-base difference. So the +3.71 W/m2 of
2xCO2 forcing reaches the ocean identically whether the flux-correction is the strong
Haney restoring (CONTROL) or the frozen q-flux + weak anomaly restoring (FREE); the
known suppression of the *response* is a response-timescale effect, not a cancellation
of the forcing term itself.
"""
import jax.numpy as jnp
import numpy as np

from chronos_esm.coupler.dino_coupling import ocean_fluxes_jax, LAT_LIN

# Myhre (1998) 2xCO2 forcing: F = 5.35 * ln(560/280) ~ 3.708 W/m2.
F_2XCO2 = 5.35 * float(np.log(560.0 / 280.0))


def _synthetic():
    """Deterministic synthetic surface fields on the linear grid (nlat, nlon).

    Returns (sst_K, u, v, t_air_K, q_air, precip, sst_target_K) -- the sst_target is
    a slightly perturbed copy of sst so the restoring term is non-trivial (and so the
    CONTROL vs FREE difference is a real test that it still cancels)."""
    nlat, nlon = LAT_LIN.shape[0], 8
    rng = np.random.default_rng(0)
    f = lambda s=1.0, b=0.0: jnp.asarray((b + s * rng.standard_normal((nlat, nlon))).astype(np.float32))
    sst = f(5, 290)
    u, v = f(5), f(5)
    ta = f(5, 288)
    qa = jnp.abs(f(1e-3))
    pr = jnp.abs(f(1e-4))
    sst_target = sst + jnp.asarray((0.5 * rng.standard_normal((nlat, nlon))).astype(np.float32))
    return sst, u, v, ta, qa, pr, sst_target


def test_co2_forcing_survives_strong_restoring_control():
    """CONTROL config: sst_target set, q_flux=None, short tau (30 d) -> strong Haney
    restoring. The forced-minus-base net heat must still equal the full 2xCO2 forcing,
    because the (co2-independent) restoring term is identical in both calls and cancels."""
    sst, u, v, ta, qa, pr, tgt = _synthetic()
    common = dict(ocean_mask=None, sst_target=tgt, restore_tau_days=30.0)
    base = ocean_fluxes_jax(sst, u, v, ta, qa, pr, co2_ppm=280.0, **common)[0]
    forced = ocean_fluxes_jax(sst, u, v, ta, qa, pr, co2_ppm=560.0, **common)[0]
    diff = float(jnp.mean(forced - base))
    np.testing.assert_allclose(diff, F_2XCO2, atol=0.05)
    # net_heat is index [0] of (net_heat, fw, tau_x, tau_y); both finite.
    assert bool(jnp.all(jnp.isfinite(base))) and bool(jnp.all(jnp.isfinite(forced)))


def test_co2_forcing_survives_qflux_free():
    """FREE config: q_flux=zeros + long tau (3650 d) -> frozen q-flux + weak anomaly
    restoring. The forced-minus-base difference must be the SAME ~3.71 W/m2 -- proving
    the forcing term survives identically; suppression of the response is a timescale
    effect, not a cancellation of the forcing."""
    sst, u, v, ta, qa, pr, tgt = _synthetic()
    nlat, nlon = LAT_LIN.shape[0], 8
    common = dict(ocean_mask=None, sst_target=tgt, restore_tau_days=3650.0,
                  q_flux=jnp.zeros((nlat, nlon)))
    base = ocean_fluxes_jax(sst, u, v, ta, qa, pr, co2_ppm=280.0, **common)[0]
    forced = ocean_fluxes_jax(sst, u, v, ta, qa, pr, co2_ppm=560.0, **common)[0]
    diff = float(jnp.mean(forced - base))
    np.testing.assert_allclose(diff, F_2XCO2, atol=0.05)
    assert bool(jnp.all(jnp.isfinite(base))) and bool(jnp.all(jnp.isfinite(forced)))


def test_control_and_free_forcing_increment_agree():
    """The forced-minus-base increment is INDEPENDENT of the flux-correction mode:
    CONTROL (strong restoring) and FREE (q-flux + weak) deliver the identical forcing
    to the ocean. This is the core P2 claim: the forcing is live and uncancelled."""
    sst, u, v, ta, qa, pr, tgt = _synthetic()
    nlat, nlon = LAT_LIN.shape[0], 8

    ctrl = dict(ocean_mask=None, sst_target=tgt, restore_tau_days=30.0)
    d_ctrl = float(jnp.mean(
        ocean_fluxes_jax(sst, u, v, ta, qa, pr, co2_ppm=560.0, **ctrl)[0]
        - ocean_fluxes_jax(sst, u, v, ta, qa, pr, co2_ppm=280.0, **ctrl)[0]))

    free = dict(ocean_mask=None, sst_target=tgt, restore_tau_days=3650.0,
                q_flux=jnp.zeros((nlat, nlon)))
    d_free = float(jnp.mean(
        ocean_fluxes_jax(sst, u, v, ta, qa, pr, co2_ppm=560.0, **free)[0]
        - ocean_fluxes_jax(sst, u, v, ta, qa, pr, co2_ppm=280.0, **free)[0]))

    np.testing.assert_allclose(d_ctrl, d_free, atol=1e-4)
    np.testing.assert_allclose(d_ctrl, F_2XCO2, atol=0.05)
