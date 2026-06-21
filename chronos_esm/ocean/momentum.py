"""Prototype prognostic BAROCLINIC momentum core (P3 / S4).

The piece that makes the AMOC physical: a time-integrated du/dt, dv/dt with Coriolis, the
hydrostatic pressure-gradient force from density, and friction/viscosity -- replacing the
algebraic thermal-wind inversion in veros_driver. Together with the S2-S3 barotropic
vorticity core it lets density DRIVE an overturning (so d(AMOC)/d(density) becomes nonzero
for the right dynamical reason, not via the interim Stommel/box closure), which is what
turns the noisy P4 bistability into a clean bifurcation.

Standalone -- NOT wired into veros_driver/main (zero regression). Validated against
geostrophic / thermal-wind balance.

Key numerics:
* SEMI-IMPLICIT Coriolis (a 2x2 rotation solve). Explicit f x u is unconditionally
  unstable; the implicit form divides by 1 + (dt f)^2 and is stable for any dt f.
* Hydrostatic pressure p(z) = g * cumsum(rho dz) from the surface down; the baroclinic
  pressure-gradient acceleration is -(1/rho0) grad p.
Fully differentiable (jnp + lax.scan).
"""
import jax
import jax.numpy as jnp

G = 9.81
RHO0 = 1025.0


def hydrostatic_pressure(rho, dz, g=G):
    """p(z) = g * cumsum(rho * dz) from the surface down (Pa, up to a surface constant).
    rho: (nz, ny, nx); dz: (nz,)."""
    dz = jnp.asarray(dz)
    return g * jnp.cumsum(rho * dz[:, None, None], axis=0)


def pressure_gradient_accel(p, dx, dy, rho0=RHO0, mask=None):
    """Horizontal PGF acceleration a = -(1/rho0) grad p. Periodic in x, one-sided in y."""
    px = (jnp.roll(p, -1, axis=2) - jnp.roll(p, 1, axis=2)) / (2.0 * dx)
    pn = jnp.concatenate([p[:, 1:, :], p[:, -1:, :]], axis=1)
    ps = jnp.concatenate([p[:, :1, :], p[:, :-1, :]], axis=1)
    py = (pn - ps) / (2.0 * dy)
    ax, ay = -px / rho0, -py / rho0
    if mask is not None:
        ax, ay = ax * mask, ay * mask
    return ax, ay


def _lap_h(field, dx, dy):
    """Horizontal Laplacian (periodic x, Dirichlet/zero-pad y) for viscosity."""
    d2x = (jnp.roll(field, -1, axis=2) - 2 * field + jnp.roll(field, 1, axis=2)) / dx ** 2
    fn = jnp.concatenate([field[:, 1:, :], jnp.zeros_like(field[:, :1, :])], axis=1)
    fs = jnp.concatenate([jnp.zeros_like(field[:, :1, :]), field[:, :-1, :]], axis=1)
    d2y = (fn - 2 * field + fs) / dy ** 2
    return d2x + d2y


def coriolis_semi_implicit(u, v, Fu, Fv, f, dt):
    """Solve the implicit Coriolis update
        (u' - u)/dt = f v' + Fu ,  (v' - v)/dt = -f u' + Fv
    as a 2x2 rotation: stable for any dt*f. f may be scalar or broadcastable."""
    ru = u + dt * Fu
    rv = v + dt * Fv
    d = 1.0 + (dt * f) ** 2
    u_new = (ru + dt * f * rv) / d
    v_new = (rv - dt * f * ru) / d
    return u_new, v_new


def step_momentum(u, v, rho, *, f, dx, dy, dz, dt, r, nu=0.0, mask=None, rho0=RHO0):
    """One time step of prognostic baroclinic momentum (semi-implicit Coriolis,
    hydrostatic PGF from rho, linear drag r, optional horizontal viscosity nu)."""
    p = hydrostatic_pressure(rho, dz)
    ax, ay = pressure_gradient_accel(p, dx, dy, rho0, mask)
    Fu = ax - r * u
    Fv = ay - r * v
    if nu > 0.0:
        Fu = Fu + nu * _lap_h(u, dx, dy)
        Fv = Fv + nu * _lap_h(v, dx, dy)
    u_new, v_new = coriolis_semi_implicit(u, v, Fu, Fv, f, dt)
    if mask is not None:
        u_new, v_new = u_new * mask, v_new * mask
    return u_new, v_new


def spin_up(rho, *, f, dx, dy, dz, dt, r, n_steps, nu=0.0, mask=None, rho0=RHO0):
    """Integrate momentum from rest under a fixed density field to (near) steady state.
    Returns (u, v). Differentiable via lax.scan."""
    u = jnp.zeros_like(rho)
    v = jnp.zeros_like(rho)

    def body(carry, _):
        u, v = carry
        u, v = step_momentum(u, v, rho, f=f, dx=dx, dy=dy, dz=dz, dt=dt, r=r,
                             nu=nu, mask=mask, rho0=rho0)
        return (u, v), None

    (u, v), _ = jax.lax.scan(body, (u, v), None, length=n_steps)
    return u, v


__all__ = ["hydrostatic_pressure", "pressure_gradient_accel", "coriolis_semi_implicit",
           "step_momentum", "spin_up", "G", "RHO0"]
