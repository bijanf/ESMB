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


__all__ = ["thc_overturning_velocity"]
