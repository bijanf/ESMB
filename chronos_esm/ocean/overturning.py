"""Thermohaline overturning closure (P3 / S1): give DENSITY a pathway to a coherent
Atlantic overturning, so d(AMOC@26.5N)/d(density) becomes nonzero and sign-correct.

The diagnostic-velocity ocean's thermal wind is local-in-latitude and carries no net
overturning to 26.5N (verified: d(AMOC)/d(subpolar salt) ~ 0). As an INTERIM mechanistic
closure -- a Stommel/Marotzke box-model-style parameterization (prescribed vertical
SHAPE, but physical STRENGTH and SENSITIVITY) -- we add a depth-integral-zero meridional
overturning velocity over the Atlantic whose strength scales with the subpolar-minus-
subtropical upper-ocean density contrast: denser subpolar water (colder/saltier) ->
stronger AMOC. Depth-integral-zero so it carries no net meridional transport (survives
the barotropic corrector) and only drives the overturning (and its w / tracer transport).

This is NOT prognostic momentum (that is P3 S2-S5: the variable-coefficient JEBAR
barotropic-vorticity solver + du/dt core). It is the cheapest differentiable change that
unblocks density-responsiveness and AMOC tipping experiments. Disable with k_vel=0.
"""
import jax
import jax.numpy as jnp
import numpy as np


def _atlantic_mask(ny, nx):
    """Atlantic basin (lon 280-360E | 0-20E, lat 34S-80N) -- matches diagnostics."""
    lon = np.linspace(0, 360, nx, endpoint=False)
    lat = np.linspace(-90, 90, ny)
    lon_m = (lon >= 280.0) | (lon <= 20.0)
    lat_m = (lat >= -34.0) & (lat <= 80.0)
    return jnp.asarray((lat_m[:, None] & lon_m[None, :]).astype(np.float64))


def thc_overturning_velocity(rho, dz, maskC, *, k_vel, drho_scale=0.1,
                             upper_depth_m=1100.0, subpolar=(45.0, 65.0),
                             subtropical=(10.0, 30.0)):
    """Depth-integral-zero Atlantic overturning velocity v_thc [m/s] (nz,ny,nx).

    strength ~ k_vel * softplus(drho / drho_scale), with drho = <rho>_subpolar,upper
    - <rho>_subtropical,upper (in-situ density [kg/m^3]). Returns (v_thc, drho, amp).
    """
    nz, ny, nx = rho.shape
    dz = jnp.asarray(dz)
    atl = _atlantic_mask(ny, nx).astype(rho.dtype)            # (ny,nx)
    lat = jnp.linspace(-90.0, 90.0, ny)

    depth_mid = jnp.cumsum(dz) - 0.5 * dz                     # (nz,) cell-centre depth
    upper = (depth_mid <= upper_depth_m).astype(rho.dtype)    # (nz,)
    H = jnp.sum(dz)
    up_d = jnp.sum(upper * dz)
    dn_d = H - up_d + 1e-20
    # vertical shape: +1 in the upper limb, negative below, depth-integral (dz-wtd) = 0.
    G = upper - (1.0 - upper) * (up_d / dn_d)                 # (nz,)

    # subpolar-minus-subtropical upper-ocean Atlantic density contrast
    upper_atl = atl[None, :, :] * maskC * upper[:, None, None]   # (nz,ny,nx)

    def band_rho(lo, hi):
        bm = ((lat >= lo) & (lat <= hi)).astype(rho.dtype)[None, :, None] * upper_atl
        return jnp.sum(rho * bm) / (jnp.sum(bm) + 1e-20)

    drho = band_rho(*subpolar) - band_rho(*subtropical)      # scalar [kg/m^3]
    amp = k_vel * jax.nn.softplus(drho / drho_scale)         # scalar [m/s], >= 0
    v_thc = amp * G[:, None, None] * atl[None, :, :] * maskC
    return v_thc, drho, amp


def subpolar_hosing_salt_tendency(hosing_sv, dx, dy, dz0, surf2d, *,
                                  band=(45.0, 65.0), s_ref=35.0):
    """Surface-salinity tendency [psu/s] from a freshwater HOSING of ``hosing_sv``
    Sverdrups spread over the subpolar (45-65N) N. Atlantic surface -- the classic
    AMOC bifurcation forcing. Freshens (negative dS) -> lighter subpolar water ->
    weaker AMOC via the thermohaline closure (the salt-advection feedback then either
    damps or amplifies it). Differentiable in hosing_sv. ``hosing_sv=0`` is a no-op.

    dx: (ny,1) zonal grid spacing [m]; dy: scalar [m]; dz0: surface layer thickness [m];
    surf2d: (ny,nx) surface wet mask.
    """
    ny, nx = surf2d.shape
    atl = _atlantic_mask(ny, nx).astype(surf2d.dtype)
    lat = jnp.linspace(-90.0, 90.0, ny)
    bandm = ((lat >= band[0]) & (lat <= band[1])).astype(surf2d.dtype)[:, None]
    region = atl * bandm * surf2d                          # (ny,nx) subpolar Atl surface
    area = jnp.sum(region * dx * dy) + 1e-20               # [m^2]
    # freshwater volume rate hosing_sv*1e6 m^3/s over `area`, depth dz0 -> virtual
    # salt removal rate; dS/dt = -(Q_fw * S_ref) / (area * dz0).
    return -(hosing_sv * 1.0e6 * s_ref) / (area * dz0) * region


__all__ = ["thc_overturning_velocity", "subpolar_hosing_salt_tendency"]
