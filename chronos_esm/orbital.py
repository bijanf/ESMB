"""Orbital (Milankovitch) insolation forcing — the P5 paleo driver.

Daily-mean top-of-atmosphere insolation as a function of the three orbital parameters
that set Earth's seasonal solar geometry:

  * obliquity   ε   — axial tilt (sets the amplitude of the seasonal cycle),
  * eccentricity e  — orbit ellipticity (modulates Earth–Sun distance over the year),
  * longitude of perihelion ϖ — the SEASON at which Earth is closest to the Sun
    (precession; sets which hemisphere's summer is intensified).

Parameterised by **solar longitude** λ (the unambiguous season coordinate: λ=0 at the NH
vernal equinox, 90° = NH summer solstice, 180° = NH autumn equinox, 270° = NH winter
solstice) rather than calendar day, so no Kepler inversion is needed for a fixed-season
experiment. `solar_longitude_from_day` provides the calendar↔λ map (Kepler) for a future
seasonal-cycle run, and `annual_mean_insolation` does the time-correct (Kepler 2nd-law)
annual average.

Standard daily-mean insolation (Berger 1978):
    sin δ = sin ε · sin λ                              (declination)
    ν = λ − ϖ                                          (true anomaly)
    (a/r)² = [(1 + e cos ν) / (1 − e²)]²               (inverse-square distance factor)
    cos H₀ = −tan φ tan δ   (clipped → polar day/night)
    S = (S₀/π) (a/r)² [H₀ sin φ sin δ + cos φ cos δ sin H₀]

Returns TOA flux (W/m²) — callers apply their own albedo / SW transmission. Pure JAX and
differentiable in every orbital parameter (the point of a differentiable paleo model:
d(climate)/d(obliquity) is a single jax.grad away).
"""

from typing import NamedTuple

import jax.numpy as jnp

SOLAR_CONSTANT = 1361.0  # W/m^2 (present-day TSI)


class OrbitalParams(NamedTuple):
    """Earth's orbital configuration. ϖ (long_perihelion_deg) is the longitude of
    perihelion measured from the moving NH vernal equinox; perihelion occurs at λ=ϖ."""

    obliquity_deg: float
    eccentricity: float
    long_perihelion_deg: float


# PMIP4 protocol values (Otto-Bliesner et al. 2017, GMD). "perihelion-180°" is the PMIP
# reporting convention; ϖ = that + 180. piControl uses the 1850 CE orbit.
ORBIT_PI = OrbitalParams(
    obliquity_deg=23.459, eccentricity=0.016764, long_perihelion_deg=280.33
)  # 1850 CE (perihelion-180 = 100.33)
ORBIT_6KA = OrbitalParams(
    obliquity_deg=24.105, eccentricity=0.018682, long_perihelion_deg=180.87
)  # mid-Holocene (perihelion-180 = 0.87)
ORBIT_PRESENT = ORBIT_PI  # alias


def _distance_factor_sq(solar_longitude_deg, orbit):
    """(a/r)² = [(1 + e cos(λ−ϖ)) / (1 − e²)]² — the inverse-square Earth–Sun distance."""
    e = orbit.eccentricity
    nu = jnp.deg2rad(solar_longitude_deg - orbit.long_perihelion_deg)  # true anomaly
    return ((1.0 + e * jnp.cos(nu)) / (1.0 - e**2)) ** 2


def daily_insolation(
    lat_rad, solar_longitude_deg, orbit=ORBIT_PRESENT, solar_constant=SOLAR_CONSTANT
):
    """Daily-mean TOA insolation [W/m²] at latitude(s) `lat_rad` for season `λ` (deg).

    lat_rad: scalar or (nlat,) array of latitudes in radians.
    solar_longitude_deg: season (0=NH spring equinox, 90=NH summer solstice, ...).
    Returns the same shape as lat_rad.
    """
    lam = jnp.deg2rad(solar_longitude_deg)
    eps = jnp.deg2rad(orbit.obliquity_deg)
    decl = jnp.arcsin(jnp.sin(eps) * jnp.sin(lam))  # declination δ
    # sunset hour angle H0 (clipped for polar day/night)
    cos_h0 = jnp.clip(-jnp.tan(lat_rad) * jnp.tan(decl), -1.0, 1.0)
    h0 = jnp.arccos(cos_h0)
    dist = _distance_factor_sq(solar_longitude_deg, orbit)
    s = (
        (solar_constant / jnp.pi)
        * dist
        * (
            h0 * jnp.sin(lat_rad) * jnp.sin(decl)
            + jnp.cos(lat_rad) * jnp.cos(decl) * jnp.sin(h0)
        )
    )
    return jnp.maximum(s, 0.0)


def solar_longitude_from_day(
    day_of_year, orbit=ORBIT_PRESENT, vernal_equinox_day=80.0, n_iter=8
):
    """Map calendar day → solar longitude λ (deg) via Kepler's equation, with λ=0 fixed at
    `vernal_equinox_day`. For a seasonal-cycle run: advance day_of_year and feed λ to
    `daily_insolation`. Newton-iterated (fixed length → AD-safe). Year length 365 days.
    """
    e = orbit.eccentricity
    varpi = jnp.deg2rad(orbit.long_perihelion_deg)

    # mean longitude advances uniformly in TIME; pin it so λ=0 at the vernal equinox.
    # true anomaly at vernal equinox: ν_ve = 0 − ϖ = −ϖ
    def true_to_mean(nu):
        E = 2.0 * jnp.arctan(
            jnp.tan(nu / 2.0) * jnp.sqrt((1 - e) / (1 + e))
        )  # eccentric anom
        return E - e * jnp.sin(E)  # mean anom

    M_ve = true_to_mean(-varpi)
    frac = (day_of_year - vernal_equinox_day) / 365.0
    M = M_ve + 2.0 * jnp.pi * frac  # mean anomaly now
    # invert Kepler M = E − e sinE for E (Newton), then E → true anomaly ν → λ = ν + ϖ
    E = M
    for _ in range(n_iter):
        E = E - (E - e * jnp.sin(E) - M) / (1.0 - e * jnp.cos(E))
    nu = 2.0 * jnp.arctan(jnp.tan(E / 2.0) * jnp.sqrt((1 + e) / (1 - e)))
    lam = jnp.rad2deg(nu) + orbit.long_perihelion_deg
    return jnp.mod(lam, 360.0)


def annual_mean_insolation(
    lat_rad, orbit=ORBIT_PRESENT, solar_constant=SOLAR_CONSTANT, n=360
):
    """Time-correct annual-mean TOA insolation [W/m²]. Averages `daily_insolation` over λ
    with the Kepler 2nd-law time weight w(λ) ∝ 1/(1+e cos(λ−ϖ))² (Earth lingers near
    aphelion), so this is the true *temporal* mean, not the λ-uniform mean."""
    lam = jnp.linspace(0.0, 360.0, n, endpoint=False)
    e = orbit.eccentricity
    nu = jnp.deg2rad(lam - orbit.long_perihelion_deg)
    w = 1.0 / (1.0 + e * jnp.cos(nu)) ** 2  # dt ∝ r² ∝ 1/(1+e cosν)²
    lat = jnp.atleast_1d(lat_rad)
    S = jnp.stack(
        [daily_insolation(lat, float(L), orbit, solar_constant) for L in lam]
    )  # (n,nlat)
    return jnp.sum(S * w[:, None], axis=0) / jnp.sum(w)


__all__ = [
    "OrbitalParams",
    "ORBIT_PI",
    "ORBIT_6KA",
    "ORBIT_PRESENT",
    "SOLAR_CONSTANT",
    "daily_insolation",
    "solar_longitude_from_day",
    "annual_mean_insolation",
]
