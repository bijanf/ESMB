"""P3 / S4: validate the prognostic baroclinic momentum core against analytic balances.
Standalone numerics -- nothing in the live model is touched, so no regression risk.

Checks:
  1. Thermal wind: spun-up steady velocity has the vertical shear
        du/dz = -(g / (rho0 f)) drho/dy
     i.e. Coriolis + hydrostatic pressure-gradient force balance correctly.
  2. Semi-implicit Coriolis is stable at large dt*f (explicit f x u would blow up).
  3. Differentiability: jax.grad of the flow strength wrt the density gradient is finite.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from chronos_esm.ocean import momentum as mom  # noqa: E402

G, RHO0 = mom.G, mom.RHO0


def _density_field(nz, ny, nx, dy, a_rho):
    """rho varying in y (axis 1) with a depth-dependent profile -> depth-varying drho/dy.
    Built with jnp so it traces through a_rho (for the differentiability test)."""
    yc = jnp.asarray((np.arange(ny) + 0.5) * dy)
    Ly = ny * dy
    vprof_np = 1.0 - 2.0 * (np.arange(nz) + 0.5) / nz        # +1 surface -> -1 bottom
    vprof = jnp.asarray(vprof_np)
    rho2 = a_rho * (yc[None, :, None] / Ly - 0.5) * vprof[:, None, None]   # (nz,ny,1)
    rho = jnp.broadcast_to(rho2, (nz, ny, nx)) + RHO0
    drho_dy = a_rho / Ly * vprof_np                          # (nz,) uniform in y
    return rho, drho_dy


def test_thermal_wind_balance():
    nz, ny, nx = 12, 40, 20
    dx = dy = 5.0e4
    dz = jnp.ones(nz) * 250.0
    f = 1.0e-4
    a_rho = 2.0                                              # density anomaly scale [kg/m3]
    rho, drho_dy = _density_field(nz, ny, nx, dy, a_rho)
    u, v = mom.spin_up(rho, f=f, dx=dx, dy=dy, dz=dz, dt=3600.0, r=5.0e-6,
                       n_steps=700)
    u = np.asarray(u)
    assert np.all(np.isfinite(u))

    # model vertical shear (between levels k-1,k) vs thermal wind, interior only
    dz0 = 250.0
    shear = (u[1:, :, :] - u[:-1, :, :]) / dz0               # (nz-1, ny, nx)
    tw = -(G / (RHO0 * f)) * drho_dy[1:]                     # (nz-1,) target per level
    interior = shear[2:-2, 8:-8, :]                          # avoid z/y edges
    target = np.broadcast_to(tw[2:-2, None, None], interior.shape)
    rel = np.abs(interior - target).mean() / (np.abs(target).mean() + 1e-30)
    assert rel < 0.12, f"thermal-wind shear rel.err={rel:.3f}"


def test_coriolis_semi_implicit_stable():
    # large dt*f: explicit Euler Coriolis amplifies ~sqrt(1+(dt f)^2) per step and blows up;
    # the semi-implicit rotation stays bounded.
    nz, ny, nx = 6, 20, 12
    dx = dy = 5.0e4
    dz = jnp.ones(nz) * 500.0
    f, dt = 1.0e-4, 5.0e4                                    # dt*f = 5
    rho, _ = _density_field(nz, ny, nx, dy, 1.0)
    u, v = mom.spin_up(rho, f=f, dx=dx, dy=dy, dz=dz, dt=dt, r=1.0e-5, n_steps=400)
    u, v = np.asarray(u), np.asarray(v)
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(v))
    assert np.max(np.abs(u)) < 5.0, f"|u|max={np.max(np.abs(u)):.2f} -- unstable"


def test_momentum_differentiable():
    nz, ny, nx = 6, 16, 10
    dx = dy = 5.0e4
    dz = jnp.ones(nz) * 500.0

    def strength(a_rho):
        rho, _ = _density_field(nz, ny, nx, dy, a_rho)
        u, _ = mom.spin_up(rho, f=1.0e-4, dx=dx, dy=dy, dz=dz, dt=3600.0, r=5.0e-6,
                           n_steps=300)
        return jnp.sqrt(jnp.mean(u ** 2))
    g = jax.grad(strength)(2.0)
    assert np.isfinite(float(g)) and abs(float(g)) > 0, f"d|u|/d(a_rho)={float(g)}"


if __name__ == "__main__":
    test_thermal_wind_balance()
    test_coriolis_semi_implicit_stable()
    test_momentum_differentiable()
    print("all momentum tests passed")
