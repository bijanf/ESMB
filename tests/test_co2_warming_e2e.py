"""P2 end-to-end CO2 warming test: in the FREE (q-flux + weak-restoring) coupled
model, a forced run (2xCO2, 560 ppm) develops a warmer ocean surface than an
unforced baseline (280 ppm) over a few coupling intervals.

This is the integration counterpart to ``tests/test_forcing_integration.py``: that
test proves the +3.71 W/m2 forcing reaches the ocean heat budget; this one proves the
freed ocean (long restoring tau + frozen q-flux) actually translates that flux into a
real SST warming through the full dinosaur<->ocean<->land<->ice coupled step. It is
marked ``slow`` (builds the real model from WOA) but is sized to run in ~1-2 min on CPU.
"""

import jax.numpy as jnp
import pytest

from chronos_esm.config import OCEAN_GRID


def _ocean_sst_mean(model, cstate):
    """Ocean-area-mean surface SST [K]: ocean.temp[0] over the model's surface ocean
    mask (model.omask, a bool sea/land mask on the linear grid)."""
    sst = cstate.ocean.temp[0]
    m = model.omask
    return float(jnp.sum(jnp.where(m, sst, 0.0)) / jnp.sum(m))


@pytest.mark.slow
def test_co2_2x_warms_ocean_surface_free_mode():
    from chronos_esm.coupler.dino_step import DinoCoupledModel

    nlat, nlon = OCEAN_GRID.nlat, OCEAN_GRID.nlon
    # FREE / forcing-responsive config: frozen q-flux (zeros here) + weak long-tau
    # anomaly restoring, so SST can respond to the imposed CO2 forcing.
    model = DinoCoupledModel(
        ocean_ic="woa",
        q_flux=jnp.zeros((nlat, nlon)),
        restore_to_woa=True,
        restore_tau_days=3650.0,
    )

    cs0 = model.init_state()
    base = cs0
    forced = cs0
    n_intervals = 5
    for _ in range(n_intervals):
        base = model.step_fast(base, co2_ppm=280.0)  # no forcing (F=0 at 280 ppm)
        forced = model.step_fast(forced, co2_ppm=560.0)  # 2xCO2 -> +3.71 W/m2

    sst_base = _ocean_sst_mean(model, base)
    sst_forced = _ocean_sst_mean(model, forced)

    # both states must be finite (no blow-up over the coupled steps)
    assert bool(jnp.all(jnp.isfinite(base.ocean.temp)))
    assert bool(jnp.all(jnp.isfinite(forced.ocean.temp)))

    # the forced run must be strictly warmer at the ocean surface
    assert sst_forced > sst_base, (
        f"2xCO2 forced ocean SST ({sst_forced:.6f} K) should exceed baseline "
        f"({sst_base:.6f} K) after {n_intervals} intervals; "
        f"diff={sst_forced - sst_base:.6e} K"
    )
