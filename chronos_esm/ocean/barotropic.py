"""Prototype prognostic BAROTROPIC vorticity core (P3 / S2).

This is the first validated building block toward replacing the diagnostic thermal-wind
velocities + the interim Stommel/box THC closure (overturning.py) with a genuinely
time-integrated overturning. It is NOT yet wired into the live ocean (veros_driver); it
lives standalone so it can be validated against analytic gyre balances with zero risk to
the running model.

Prognostic relative vorticity zeta with the streamfunction psi diagnosed each step from
the AD-safe variable-coefficient elliptic invert (solver.solve_elliptic_varcoef):

    d zeta/dt = -beta * dpsi/dx - r * zeta + F + nu * lap(zeta),
    zeta = div(coef * grad psi),   u = -dpsi/dy,  v = dpsi/dx,

with F = curl(tau)/(rho0 * H) the wind-stress-curl forcing. coef = 1/H gives the
topographic (JEBAR-like) operator over real bathymetry; coef = 1 is the flat-bottom
Stommel limit. Steady state -> r*lap(psi) + beta*dpsi/dx = F (Stommel gyre: Sverdrup
interior + a western boundary layer of width delta = r/beta). Fully differentiable.
"""
import jax
import jax.numpy as jnp

from chronos_esm.ocean.solver import apply_elliptic_varcoef, solve_elliptic_varcoef


def ddx_centered(p, dx):
    """Centred d/dx, periodic in x."""
    return (jnp.roll(p, -1, axis=1) - jnp.roll(p, 1, axis=1)) / (2.0 * dx)


def ddy_centered(p, dy):
    """Centred d/dy, one-sided at the y-edges (Dirichlet/closed)."""
    pn = jnp.concatenate([p[1:, :], p[-1:, :]], axis=0)
    ps = jnp.concatenate([p[:1, :], p[:-1, :]], axis=0)
    return (pn - ps) / (2.0 * dy)


def wind_stress_curl(taux, tauy, dx, dy):
    """curl(tau) = d(tauy)/dx - d(taux)/dy."""
    return ddx_centered(tauy, dx) - ddy_centered(taux, dy)


def velocities_from_psi(psi, dx, dy, mask=None):
    """u = -dpsi/dy, v = dpsi/dx (barotropic transport per unit depth)."""
    u = -ddy_centered(psi, dy)
    v = ddx_centered(psi, dx)
    if mask is not None:
        u, v = u * mask, v * mask
    return u, v


def step_barotropic(zeta, psi, F, *, dx, dy, dt, beta, r, mask, coef,
                    nu=0.0, max_iter=400, tol=1e-8):
    """One forward-Euler step of the linear barotropic vorticity equation.

    Prognose zeta, then diagnose psi by inverting zeta = div(coef * grad psi).
    Returns (zeta_new, psi_new).
    """
    dpsidx = ddx_centered(psi, dx) * mask
    tend = (-beta * dpsidx - r * zeta + F) * mask
    if nu > 0.0:
        tend = tend + nu * apply_elliptic_varcoef(zeta, jnp.ones_like(coef), dx, dy, mask)
    zeta_new = (zeta + dt * tend) * mask
    psi_new, _ = solve_elliptic_varcoef(coef, zeta_new, dx, dy, mask=mask, x0=psi,
                                        max_iter=max_iter, tol=tol)
    return zeta_new, psi_new


def spin_up_gyre(F, *, dx, dy, dt, beta, r, mask, coef, n_steps, nu=0.0,
                 max_iter=400):
    """Integrate the barotropic gyre to (near) steady state from rest. Returns
    (psi, zeta) using a differentiable lax.scan loop."""
    ny, nx = F.shape
    z0 = jnp.zeros((ny, nx), F.dtype)
    p0 = jnp.zeros((ny, nx), F.dtype)

    def body(carry, _):
        z, p = carry
        z, p = step_barotropic(z, p, F, dx=dx, dy=dy, dt=dt, beta=beta, r=r,
                               mask=mask, coef=coef, nu=nu, max_iter=max_iter)
        return (z, p), None

    (zeta, psi), _ = jax.lax.scan(body, (z0, p0), None, length=n_steps)
    return psi, zeta


__all__ = ["wind_stress_curl", "velocities_from_psi", "step_barotropic",
           "spin_up_gyre", "ddx_centered", "ddy_centered"]
