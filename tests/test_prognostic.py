"""S5c: the assembled prognostic ocean dynamics (chronos_esm/ocean/prognostic.py).

Network-free synthetic checks of the STABLE baroclinic path: density drives a
finite, bounded overturning through the assembled step, and the spin-up is
differentiable end to end. (The wind-driven barotropic mode on the full global grid
needs further CFL/conditioning work -- see the module docstring; not asserted here.)
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from chronos_esm.ocean import prognostic  # noqa: E402

A = 6.371e6
NZ, NY, NX = 6, 24, 48
LAT = jnp.asarray(np.deg2rad(np.linspace(-88.0, 88.0, NY)))
DLON = 2.0 * np.pi / NX
DLAT = float(LAT[1] - LAT[0])
DZ = jnp.asarray(np.full(NZ, 400.0))


def _density():
    latg = LAT[None, :, None]
    depth = jnp.cumsum(DZ)[:, None, None]
    rho = 1025.0 + 0.8 * jnp.abs(jnp.sin(latg)) + 0.4 * (depth / depth.max())
    return jnp.broadcast_to(rho, (NZ, NY, NX)).astype(jnp.float64)


def _mask():
    m = np.ones((NZ, NY, NX))
    m[:, :, 0] = 0.0  # western wall
    return jnp.asarray(m)


def test_prognostic_baroclinic_stable_and_bounded():
    """Wind off -> the barotropic mode stays zero; the density-driven baroclinic
    overturning is finite and bounded (no blow-up), with realistic interior speeds."""
    rho = _density()
    mask = _mask()
    zero = jnp.zeros((NY, NX))
    u, v, psi, zeta = prognostic.spin_up_prognostic_dynamics(
        rho,
        zero,
        zero,
        lat=LAT,
        dlon=DLON,
        dlat=DLAT,
        a=A,
        dz=DZ,
        dt=1800.0,
        A_h_bc=5.0e6,
        A_h_bt=5.0e6,
        mask3d=mask,
        n_steps=40,
        visc_iter=100,
    )
    assert bool(jnp.isfinite(u).all() and jnp.isfinite(v).all())
    assert float(jnp.max(jnp.abs(psi))) == 0.0  # no wind -> no barotropic gyre
    assert float(jnp.abs(v).max()) < 5.0  # bounded, realistic interior speed
    # the baroclinic flow has depth structure (an overturning), not just a slab
    H = prognostic.column_depth(DZ, mask)  # floored -> no /0 at land columns
    vbar = jnp.sum(v * DZ[:, None, None] * mask, axis=0) / H
    bc = (v - vbar[None]) * mask  # baroclinic part on wet cells only
    assert float(jnp.std(bc)) > 0.0


def test_prognostic_differentiable():
    """d(circulation)/d(density) flows through the assembled spin-up."""
    rho0 = _density()
    mask = _mask()
    zero = jnp.zeros((NY, NX))
    anom = jnp.abs(jnp.sin(LAT))[None, :, None] * jnp.ones((NZ, NY, NX))

    def loss(eps):
        _, v, _, _ = prognostic.spin_up_prognostic_dynamics(
            rho0 + eps * anom,
            zero,
            zero,
            lat=LAT,
            dlon=DLON,
            dlat=DLAT,
            a=A,
            dz=DZ,
            dt=1800.0,
            A_h_bc=5.0e6,
            A_h_bt=5.0e6,
            mask3d=mask,
            n_steps=6,
            visc_iter=60,
        )
        return jnp.mean(v**2)

    g = float(jax.grad(loss)(0.0))
    assert np.isfinite(g) and g != 0.0


if __name__ == "__main__":
    test_prognostic_baroclinic_stable_and_bounded()
    test_prognostic_differentiable()
    print("all S5c prognostic-dynamics tests passed")
