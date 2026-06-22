"""P5 paleo: the seasonal cycle wired into the active dino coupling.

The active model was perpetual-equinox (insolation hardcoded at day=80). DinoCoupledModel
now takes seasonal=True + an orbit (chronos_esm.orbital) and recomputes insolation each
interval from the model day, so it has a real seasonal cycle and a 6 ka orbit drives the
mid-Holocene signal. The orbital forcing itself is validated in tests/test_orbital.py;
here we check the WIRING:
  1. seasonal insolation varies with the day correctly (NH summer > NH winter in the NH);
  2. it uses the model's orbit + the surface-SW convention (matches a direct orbital call);
  3. a seasonal step_fast runs finite (the coupled model is stable under seasonal forcing).

Heavy DinoCoupledModel build is module-scoped (per the OOM-segfault gotcha). Default
(seasonal=False) is covered by tests/test_dino_step.py — that path is untouched.
"""
import numpy as np
import jax.numpy as jnp
import pytest

from chronos_esm import orbital
from chronos_esm.atmos import physics as aphys
from chronos_esm.config import OCEAN_GRID

LATD = np.linspace(-90, 90, OCEAN_GRID.nlat)


def _at(arr, deg, axis_mean=True):
    a = np.asarray(arr)
    if a.ndim == 2 and axis_mean:
        a = a.mean(axis=1)
    return float(a[int(np.argmin(np.abs(LATD - deg)))])


@pytest.fixture(scope="module")
def model():
    from chronos_esm.coupler.dino_step import DinoCoupledModel
    return DinoCoupledModel(seasonal=True, orbit=orbital.ORBIT_PI, interval=1.0)


# day=80 is the vernal equinox; +91 ~ NH summer solstice, +273 ~ NH winter solstice
SUMMER_DAY = 80.0 + 91.0
WINTER_DAY = 80.0 + 273.0


def test_seasonal_insolation_varies_with_day(model):
    sw_summer, _ = model._insolation(jnp.asarray(SUMMER_DAY))
    sw_winter, _ = model._insolation(jnp.asarray(WINTER_DAY))
    # NH gets much more sun at the NH summer solstice than at NH winter
    assert _at(sw_summer, 45) > _at(sw_winter, 45) + 50.0, "no NH seasonal swing"
    # SH is the opposite sign
    assert _at(sw_summer, -45) < _at(sw_winter, -45), "SH season not reversed"


def test_insolation_uses_orbit_and_convention(model):
    """_insolation must return the SURFACE SW = orbital-TOA(orbit) * (1-albedo) * 0.60 for
    BOTH the land/ice field and the ocean override (the same convention compute_solar_insolation
    used, so the ocean's own *(1-ALBEDO_OCEAN) is applied exactly once -- not the raw TOA, which
    would double-count and ~2x the ocean SW)."""
    day = jnp.asarray(SUMMER_DAY)
    sw, insol_override = model._insolation(day)
    lam = orbital.solar_longitude_from_day(jnp.mod(day, 365.0), orbital.ORBIT_PI)
    ref_toa = orbital.daily_insolation(model.lat_rad, lam, orbital.ORBIT_PI)[:, None]
    ref_sw = ref_toa * (1.0 - aphys.compute_albedo(model.lat_rad)[:, None]) * 0.60
    # the ocean override is the surface SW (NOT raw TOA): guard against the 2x-SW bug
    assert np.allclose(np.asarray(insol_override), np.asarray(ref_sw), rtol=1e-6)
    assert np.allclose(np.asarray(sw)[:, :1], np.asarray(ref_sw), rtol=1e-6)
    # surface SW must be well below raw TOA (the bug made them equal)
    assert float(np.asarray(insol_override).max()) < 0.75 * float(np.asarray(ref_toa).max())


def test_seasonal_step_runs_finite(model):
    st = model.init_state()
    st = model.step_fast(st, co2_ppm=280.0)
    st = model.step_fast(st, co2_ppm=280.0)
    assert bool(jnp.isfinite(st.ocean.temp).all())
    assert bool(jnp.isfinite(st.atmos.vorticity).all())


if __name__ == "__main__":
    from chronos_esm.coupler.dino_step import DinoCoupledModel
    m = DinoCoupledModel(seasonal=True, orbit=orbital.ORBIT_PI)
    test_seasonal_insolation_varies_with_day(m)
    test_insolation_uses_orbit_and_convention(m)
    test_seasonal_step_runs_finite(m)
    sw_s, _ = m._insolation(jnp.asarray(SUMMER_DAY))
    sw_w, _ = m._insolation(jnp.asarray(WINTER_DAY))
    print("NH 45N surface SW: summer %.0f  winter %.0f W/m^2" % (_at(sw_s, 45), _at(sw_w, 45)))
    print("all seasonal-cycle wiring tests passed")
