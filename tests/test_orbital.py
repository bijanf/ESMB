"""P5 paleo: orbital (Milankovitch) insolation forcing.

Validates chronos_esm/orbital.py against textbook / PMIP4 numbers BEFORE it drives the
coupled model, so the paleo forcing itself is trustworthy (a model-independent physical
check):
  1. present-day NH summer-solstice insolation has the right magnitude (North-Pole polar-day
     value ~520 W/m^2, the classic Milankovitch number);
  2. the mid-Holocene (6 ka) signal: NH summer insolation INCREASES by ~+20-30 W/m^2 at
     mid-high NH latitudes and NH winter DECREASES (the seasonal amplification that drives
     the enhanced Holocene monsoon) -- the canonical PMIP4 anomaly;
  3. orbital forcing is a seasonal REDISTRIBUTION: the true annual-global mean barely changes
     (<0.5 W/m^2; it depends only on eccentricity);
  4. the forcing is differentiable in the orbital parameters (d insolation / d obliquity
     finite and nonzero) -- the point of a differentiable paleo model.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from chronos_esm.orbital import ORBIT_6KA  # noqa: E402
from chronos_esm.orbital import (
    ORBIT_PI,
    OrbitalParams,
    annual_mean_insolation,
    daily_insolation,
    solar_longitude_from_day,
)

LAT = jnp.deg2rad(jnp.linspace(-90, 90, 49))  # incl. poles for the polar-day check
latd = np.linspace(-90, 90, 49)


def _at(arr, deg):
    return float(np.asarray(arr)[int(np.argmin(np.abs(latd - deg)))])


def test_present_day_summer_solstice_magnitude():
    """NH summer solstice (lambda=90): North Pole sees polar-day insolation ~520 W/m^2,
    and the NH is sunlit more than the SH."""
    S = daily_insolation(LAT, 90.0, ORBIT_PI)
    npole = _at(S, 90)
    assert (
        480.0 < npole < 560.0
    ), f"N-pole summer-solstice insol {npole:.0f} (expect ~520)"
    # SH pole is in polar night -> zero
    assert _at(S, -90) < 1.0
    # equator gets substantial but less than the summer pole at solstice
    assert _at(S, 0) > 350.0


def test_midholocene_summer_increase():
    """The canonical mid-Holocene signal: 6 ka NH summer insolation is HIGHER than PI
    by ~+20-30 W/m^2 at mid-high NH latitudes (stronger tilt + perihelion in NH summer).
    """
    dS = np.asarray(
        daily_insolation(LAT, 90.0, ORBIT_6KA) - daily_insolation(LAT, 90.0, ORBIT_PI)
    )
    # positive across the NH summer hemisphere
    nh = latd >= 10
    assert np.all(
        dS[nh] > 0.0
    ), "6ka NH summer insolation should exceed PI everywhere in NH"
    # magnitude at 45N is the textbook ~+20-30 W/m^2
    d45 = _at(dS, 45)
    assert (
        15.0 < d45 < 40.0
    ), f"6ka-PI JJA anomaly at 45N = {d45:+.1f} W/m^2 (expect ~+25)"


def test_midholocene_winter_decrease():
    """Seasonal amplification: 6 ka NH WINTER (lambda=270) insolation is LOWER than PI in
    the NH -- the other half of the redistribution."""
    dS = np.asarray(
        daily_insolation(LAT, 270.0, ORBIT_6KA) - daily_insolation(LAT, 270.0, ORBIT_PI)
    )
    # NH (sunlit part in winter is the low-lat NH) decreases; check 0-30N where there is sun
    band = (latd >= 0) & (latd <= 30)
    assert np.all(dS[band] < 0.0), "6ka NH winter insolation should be below PI"


def test_annual_global_mean_nearly_unchanged():
    """Orbital forcing redistributes insolation seasonally/latitudinally; the global ANNUAL
    mean changes only via eccentricity and is tiny (<0.5 W/m^2)."""
    coslat = np.cos(latd * np.pi / 180.0)
    pi_am = np.asarray(annual_mean_insolation(LAT, ORBIT_PI))
    ho_am = np.asarray(annual_mean_insolation(LAT, ORBIT_6KA))
    g_pi = np.sum(pi_am * coslat) / np.sum(coslat)
    g_ho = np.sum(ho_am * coslat) / np.sum(coslat)
    assert (
        abs(g_ho - g_pi) < 0.5
    ), f"global-annual-mean changed {g_ho - g_pi:+.2f} W/m^2 (expect ~0)"
    # global annual mean is ~ S0/4 (geometry): ~340 W/m^2
    assert 330.0 < g_pi < 350.0, f"global-annual-mean insol {g_pi:.1f} (expect ~340)"


def test_kepler_roundtrip_equinox():
    """lambda(vernal-equinox-day) == 0 by construction; solstice day maps near lambda=90."""
    lam_ve = float(solar_longitude_from_day(80.0, ORBIT_PI, vernal_equinox_day=80.0))
    assert min(lam_ve, 360 - lam_ve) < 1.0, f"lambda at VE day = {lam_ve} (expect ~0)"
    # ~3 months after VE -> near summer solstice
    lam_ss = float(
        solar_longitude_from_day(80.0 + 365.0 / 4.0, ORBIT_PI, vernal_equinox_day=80.0)
    )
    assert 80.0 < lam_ss < 100.0, f"lambda ~1/4 yr after VE = {lam_ss} (expect ~90)"


def test_differentiable_in_obliquity():
    """d(NH-summer insolation)/d(obliquity) is finite and nonzero -- the differentiable
    paleo hook (more tilt -> more summer insolation at high lat)."""
    lat45 = jnp.deg2rad(45.0)

    def insol_of_obliquity(eps_deg):
        orb = OrbitalParams(
            eps_deg, ORBIT_PI.eccentricity, ORBIT_PI.long_perihelion_deg
        )
        return daily_insolation(lat45, 90.0, orb)

    g = float(jax.grad(insol_of_obliquity)(23.459))
    assert (
        np.isfinite(g) and g > 0.5
    ), f"d(insol)/d(obliquity) at 45N = {g} (expect clearly +)"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    dS = np.asarray(
        daily_insolation(LAT, 90.0, ORBIT_6KA) - daily_insolation(LAT, 90.0, ORBIT_PI)
    )
    print("\n6ka-PI summer-solstice anomaly (W/m^2):")
    for d in (0, 30, 45, 65, 90):
        print(f"  {d:3d}N: {_at(dS, d):+5.1f}")
    print("all orbital-forcing tests passed")
