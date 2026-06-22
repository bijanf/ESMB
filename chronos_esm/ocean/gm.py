"""Gent-McWilliams (GM90) eddy parameterization for the prognostic ocean core (P3).

WHY THIS EXISTS
---------------
At T31 (~3.75 deg) the ocean cannot resolve mesoscale eddies, so the prognostic
baroclinic momentum core (``veros_driver.step_ocean(prognostic_momentum=True)`` +
``momentum.py``) builds its overturning from the *full* (eddy-unflattened) isopycnal
slope and gives an AMOC that is 20-40x too large / reversed -- the documented
"T31 resolution barrier" (see docs/prognostic_ocean_core.md, S5). The missing physics
is the baroclinic eddies that flatten isopycnals and oppose the mean overturning.

Gent-McWilliams (1990) parameterizes those eddies as an adiabatic *eddy-induced*
(bolus) velocity that advects tracers down the isopycnal-slope gradient, flattening
isopycnals and extracting available potential energy. In the residual-mean (TEM)
framework the overturning that actually transports heat is the **residual**
circulation ``v_res = v_mean + v_eddy``; the eddy (bolus) cell opposes the mean where
isopycnals are steep, so the residual MOC is smaller -- and more physical -- than the
Eulerian-mean MOC the bare momentum core produces.

This module is the *correct*, AD-safe GM scheme (tapered, latitude-aware), superseding
the partial version in ``mixing.py`` (which has no slope/surface taper and uses a
constant ``dx``). It is wired into ``step_ocean`` behind the ``gm_on`` flag (default
off -> zero regression).

SCOPE / HONEST LIMITS
---------------------
GM cuts coarse-model spurious overturning by *tens of percent* and fixes its sign; it
does NOT by itself land a T31 AMOC on the observed ~15 Sv (the literature finds the
Atlantic cell is less GM-sensitive than the Deacon cell). Reaching the observed AMOC
needs GM + sustained integration (isopycnal flattening is a multi-decade adjustment) +
likely finer resolution. Redi isoneutral diffusion (``mixing.rotate_tensor``, currently
dead) and a spatially-varying kappa_GM are deferred.

All ops are pure ``jnp`` and differentiable (smooth tapers, sign-preserving epsilon
floor on d(rho)/dz, soft slope clip) so ``d(AMOC)/d(density)`` stays finite and nonzero.
"""

import jax.numpy as jnp

from chronos_esm.config import EARTH_RADIUS

# --- regularization / taper constants -------------------------------------------
SMALL_DRHODZ = 1e-8  # sign-preserving floor on |d rho/dz| [kg/m^4] (matches mixing.py)
S_MAX = 1e-2  # soft slope-clip asymptote [dimensionless]
S_CRIT = 4e-3  # DM95 critical slope (taper midpoint)
S_WIDTH = 1e-3  # DM95 taper width
SURFACE_TAPER_M = (
    150.0  # depth scale over which the bolus is ramped down to the lid [m]
)
POLE_FLOOR_DEG = 80.0  # cos(lat) floor latitude for the longitudinal metric


def softclip(s, s_max=S_MAX):
    """Smooth, AD-safe slope limiter -> +/- s_max asymptote (cf. mixing.slope_limiter)."""
    return s / jnp.sqrt(1.0 + (s / s_max) ** 2)


def latitude_dx(ny, nx):
    """Longitudinal grid spacing dx(lat) [m] on the lat-lon ocean grid, shape (1,ny,1).

    A constant (equatorial) dx -- as the legacy slope code uses -- understates the
    zonal density gradient toward the poles and so understates the slope GM acts on.
    Floor cos(lat) at +/-POLE_FLOOR_DEG (mirrors the atmosphere's polar-dx fix) so the
    metric does not collapse at the poles.
    """
    lat = jnp.deg2rad(jnp.linspace(-90.0, 90.0, ny))
    dlon = 2.0 * jnp.pi / nx
    cos_floor = jnp.cos(jnp.deg2rad(POLE_FLOOR_DEG))
    dx = EARTH_RADIUS * jnp.maximum(jnp.cos(lat), cos_floor) * dlon
    return dx[None, :, None]


def isopycnal_slopes(rho, dx_3d, dy, dz):
    """Isopycnal slopes S_x = -(d rho/dx)/(d rho/dz), S_y = -(d rho/dy)/(d rho/dz).

    Vertical gradient is centered (one-sided at the surface/floor); |d rho/dz| is
    floored with a sign-preserving epsilon (stable stratification has d rho/dz > 0)
    so the division is AD-safe. Slopes are soft-clipped to +/-S_MAX. dx_3d is the
    latitude-dependent metric from ``latitude_dx``; dy is a scalar.
    """
    dz_3d = dz.reshape(-1, 1, 1)
    drho_dx = (jnp.roll(rho, -1, axis=2) - jnp.roll(rho, 1, axis=2)) / (2.0 * dx_3d)
    rho_pad = jnp.pad(rho, ((0, 0), (1, 1), (0, 0)), mode="edge")
    drho_dy = (rho_pad[:, 2:, :] - rho_pad[:, :-2, :]) / (2.0 * dy)

    dist = 0.5 * (dz_3d[:-1] + dz_3d[1:])
    grad_if = (rho[1:] - rho[:-1]) / dist  # at interfaces k+1/2
    drho_dz = jnp.concatenate(
        [grad_if[:1], 0.5 * (grad_if[:-1] + grad_if[1:]), grad_if[-1:]], axis=0
    )
    drho_dz = jnp.where(
        jnp.abs(drho_dz) < SMALL_DRHODZ,
        jnp.sign(drho_dz + 1e-16) * SMALL_DRHODZ,
        drho_dz,
    )
    sx = softclip(-drho_dx / drho_dz)
    sy = softclip(-drho_dy / drho_dz)
    return sx, sy


def slope_taper(sx, sy):
    """DM95 steep-slope taper: ~1 for gentle slopes, smoothly -> 0 for |S| >> S_CRIT.

    Steep isopycnals (mixed layer, fronts) would give an unbounded / unstable bolus;
    tapering kappa there is the standard stabilizer (Danabasoglu & McWilliams 1995).
    """
    smag = jnp.sqrt(sx**2 + sy**2 + 1e-24)
    return 0.5 * (1.0 + jnp.tanh((S_CRIT - smag) / S_WIDTH))


def surface_taper(dz):
    """Smooth ramp 0 near the surface -> 1 at depth (LDD97 spirit), shape (nz,1,1).

    Ramps the bolus down toward the rigid lid so the eddy streamfunction is small in
    the weakly-stratified surface layers; depth-based (not f-based) to stay
    equator-safe and AD-clean. Combined with the Psi*=0 surface boundary condition.
    """
    z_center = jnp.cumsum(dz) - 0.5 * dz  # layer-center depth [m]
    ramp = 1.0 - jnp.exp(-z_center / SURFACE_TAPER_M)
    return ramp[:, None, None]


def eddy_induced_velocity(kappa_gm, rho, dx_3d, dy, dz, maskC=None):
    """GM bolus (eddy-induced) velocity (u*, v*) at cell centers [m/s].

    u* = d/dz(kappa_eff * S_x), v* = d/dz(kappa_eff * S_y), with the bolus
    streamfunction Psi* = kappa_eff * S set to zero at the surface and floor
    interfaces (no eddy transport through the rigid lid / sea floor). kappa_eff =
    kappa_gm * slope_taper * surface_taper. Sign convention matches the legacy
    ``mixing.compute_gm_bolus_velocity`` so ``gm_on`` is a tapered drop-in.

    The depth integral of (u*, v*) is identically zero (Psi*=0 BCs) -> the bolus
    carries NO net transport; it only drives the overturning + its tracer transport.

    Args:
        kappa_gm: scalar or (1,ny,1)/(nz,ny,nx) GM diffusivity [m^2/s] (may already
            carry the pole mask, as ``kappa_gm * interior_mask_3d`` in step_ocean).
        rho: density (nz,ny,nx); dx_3d: (1,ny,1) [m]; dy: scalar [m]; dz: (nz,) [m].
        maskC: optional (nz,ny,nx) wet=1 mask; if given, faces closed at coasts/floor.
    """
    nz, ny, nx = rho.shape
    dz_3d = dz.reshape(-1, 1, 1)
    sx, sy = isopycnal_slopes(rho, dx_3d, dy, dz)
    kappa_eff = kappa_gm * slope_taper(sx, sy) * surface_taper(dz)
    fx = kappa_eff * sx
    fy = kappa_eff * sy
    if maskC is not None:
        fx = fx * maskC
        fy = fy * maskC

    fx_if = 0.5 * (fx[:-1] + fx[1:])  # interface k+1/2 values
    fy_if = 0.5 * (fy[:-1] + fy[1:])
    if maskC is not None:
        iface = maskC[:-1] * maskC[1:]  # open only if both centers wet
        fx_if = fx_if * iface
        fy_if = fy_if * iface
    z = jnp.zeros((1, ny, nx))
    fx_if = jnp.concatenate([z, fx_if, z], axis=0)  # Psi*=0 at surface & floor
    fy_if = jnp.concatenate([z, fy_if, z], axis=0)

    u_star = (fx_if[1:] - fx_if[:-1]) / dz_3d
    v_star = (fy_if[1:] - fy_if[:-1]) / dz_3d
    if maskC is not None:
        u_star = u_star * maskC
        v_star = v_star * maskC
    return u_star, v_star


__all__ = [
    "softclip",
    "latitude_dx",
    "isopycnal_slopes",
    "slope_taper",
    "surface_taper",
    "eddy_induced_velocity",
]
