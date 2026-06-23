"""S5a: the spherical, implicitly-viscous, no-slip baroclinic momentum core
(chronos_esm/ocean/momentum.py:*_sphere). Network-free synthetic checks that it

  1. solves the implicit-viscosity Helmholtz system correctly (no-slip on land);
  2. is POLE-STABLE -- the old explicit-viscosity core went NaN at the poles where
     the viscous CFL collapses; the implicit core must stay finite for any A_h;
  3. uses viscosity as a controlled lever (more A_h -> less spurious transport);
  4. is differentiable end to end (jax.grad through a spin-up).
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from chronos_esm.ocean import momentum  # noqa: E402
from chronos_esm.ocean.momentum import (  # noqa: E402
    _face_conductances,
    _lap3d,
)

A = 6.371e6  # Earth radius [m]
NZ, NY, NX = 6, 24, 48
LAT = jnp.asarray(np.deg2rad(np.linspace(-88.0, 88.0, NY)))
DLON = 2.0 * np.pi / NX
DLAT = float(LAT[1] - LAT[0])
DZ = jnp.asarray(np.full(NZ, 200.0))


def _synthetic_density():
    """A stratified ocean with a meridional surface-density gradient (denser poles)
    that drives a circulation, plus depth increase."""
    latg = LAT[None, :, None]
    depth = jnp.cumsum(DZ)[:, None, None]
    rho = (
        1025.0
        + 0.8 * jnp.abs(jnp.sin(latg))  # denser toward the poles
        + 0.4 * (depth / depth.max())  # stable stratification
        + 0.02 * jnp.cos(jnp.arange(NX))[None, None, :]  # zonal structure
    )
    return jnp.broadcast_to(rho, (NZ, NY, NX)).astype(jnp.float64)


def _mask_with_land():
    """Ocean everywhere except a meridional land wall (to test the no-slip boundary)
    and a small land block."""
    m = np.ones((NZ, NY, NX))
    m[:, :, 0] = 0.0  # a western wall at x=0
    m[:, NY // 2 : NY // 2 + 2, 10:14] = 0.0  # an interior land block
    return jnp.asarray(m)


def test_implicit_viscosity_solves_helmholtz_with_noslip():
    """(I - alpha*lap) x = field on ocean (small residual); x = 0 on land (no-slip)."""
    mask = _mask_with_land()
    rng = np.random.default_rng(0)
    field = jnp.asarray(rng.standard_normal((NZ, NY, NX))) * mask
    alpha = 5.0e3 * 1800.0  # A_h*dt, large
    x = momentum.implicit_viscosity_sphere(
        field, alpha, LAT, DLON, DLAT, A, mask, max_iter=400, tol=1e-10
    )
    assert bool(jnp.isfinite(x).all())
    # no-slip: solution is exactly zero on land
    assert float(jnp.max(jnp.abs(x * (1.0 - mask)))) == 0.0
    # residual of area*(I - alpha*lap) x - area*field is small on ocean
    cE, cW, cN, cS, area = _face_conductances(LAT, DLON, DLAT, A)
    resid = (area * x + alpha * _lap3d(x, cE, cW, cN, cS) - area * field) * mask
    rel = float(jnp.linalg.norm(resid) / jnp.linalg.norm(area * field * mask))
    assert rel < 1e-6, f"Helmholtz residual too large: {rel}"


def test_implicit_viscosity_smooths_and_is_cfl_free():
    """A huge A_h damps small scales without blowing up (implicit -> no CFL limit)."""
    mask = jnp.ones((NZ, NY, NX))
    checker = jnp.asarray(
        ((np.indices((NZ, NY, NX)).sum(0)) % 2) * 2.0 - 1.0
    )  # +/-1 checkerboard
    alpha = 1.0e8  # absurdly large; an explicit scheme would explode
    x = momentum.implicit_viscosity_sphere(
        checker, alpha, LAT, DLON, DLAT, A, mask, max_iter=400, tol=1e-10
    )
    assert bool(jnp.isfinite(x).all())
    assert float(jnp.var(x)) < float(jnp.var(checker))  # smoothed


def test_spin_up_is_pole_stable():
    """The old explicit-viscosity core went NaN at the poles. The implicit core must
    stay finite there for a viscosity that resolves a Munk layer."""
    rho = _synthetic_density()
    mask = _mask_with_land()
    A_h = 5.0e4  # m^2/s -- large coarse-ocean viscosity
    u, v = momentum.spin_up_sphere(
        rho,
        lat=LAT,
        dlon=DLON,
        dlat=DLAT,
        a=A,
        dz=DZ,
        dt=1800.0,
        A_h=A_h,
        r=1.0 / (60 * 86400),
        n_steps=40,
        mask=mask,
        visc_iter=150,
        visc_tol=1e-9,
    )
    assert bool(jnp.isfinite(u).all() and jnp.isfinite(v).all())
    # finite AND bounded (not a slow blow-up): coarse geostrophic speeds stay O(<10 m/s)
    assert float(jnp.abs(v).max()) < 10.0
    # the polar rows specifically are finite (the old failure mode)
    assert bool(jnp.isfinite(v[:, :2, :]).all() and jnp.isfinite(v[:, -2:, :]).all())


def test_viscosity_reduces_spurious_transport():
    """More lateral viscosity -> smaller spurious coarse-grid transport (the lever
    that the giant Rayleigh drag was wrongly doing)."""
    rho = _synthetic_density()
    mask = jnp.ones((NZ, NY, NX))
    kw = dict(
        lat=LAT,
        dlon=DLON,
        dlat=DLAT,
        a=A,
        dz=DZ,
        dt=1800.0,
        r=1.0 / (60 * 86400),
        n_steps=40,
        mask=mask,
        visc_iter=150,
        visc_tol=1e-9,
    )
    _, v_lo = momentum.spin_up_sphere(rho, A_h=1.0e4, **kw)
    _, v_hi = momentum.spin_up_sphere(rho, A_h=2.0e5, **kw)
    assert float(jnp.abs(v_hi).max()) < float(jnp.abs(v_lo).max())


def test_spin_up_differentiable():
    """d(circulation)/d(density perturbation) flows through the spin-up (the whole
    point: density drives the overturning, differentiably)."""
    rho0 = _synthetic_density()
    mask = jnp.ones((NZ, NY, NX))
    anom = jnp.abs(jnp.sin(LAT))[None, :, None] * jnp.ones((NZ, NY, NX))

    def loss(eps):
        rho = rho0 + eps * anom
        _, v = momentum.spin_up_sphere(
            rho,
            lat=LAT,
            dlon=DLON,
            dlat=DLAT,
            a=A,
            dz=DZ,
            dt=1800.0,
            A_h=5.0e4,
            r=1.0 / (60 * 86400),
            n_steps=8,
            mask=mask,
            visc_iter=80,
            visc_tol=1e-8,
        )
        return jnp.mean(v**2)

    g = float(jax.grad(loss)(0.0))
    assert np.isfinite(g) and g != 0.0


if __name__ == "__main__":
    test_implicit_viscosity_solves_helmholtz_with_noslip()
    test_implicit_viscosity_smooths_and_is_cfl_free()
    test_spin_up_is_pole_stable()
    test_viscosity_reduces_spurious_transport()
    test_spin_up_differentiable()
    print("all S5a spherical-momentum tests passed")
