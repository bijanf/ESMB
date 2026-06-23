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
    d2x = (jnp.roll(field, -1, axis=2) - 2 * field + jnp.roll(field, 1, axis=2)) / dx**2
    fn = jnp.concatenate([field[:, 1:, :], jnp.zeros_like(field[:, :1, :])], axis=1)
    fs = jnp.concatenate([jnp.zeros_like(field[:, :1, :]), field[:, :-1, :]], axis=1)
    d2y = (fn - 2 * field + fs) / dy**2
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
        u, v = step_momentum(
            u, v, rho, f=f, dx=dx, dy=dy, dz=dz, dt=dt, r=r, nu=nu, mask=mask, rho0=rho0
        )
        return (u, v), None

    (u, v), _ = jax.lax.scan(body, (u, v), None, length=n_steps)
    return u, v


# ===========================================================================
# SPHERICAL, IMPLICITLY-VISCOUS, NO-SLIP baroclinic momentum core (P3 / S5a).
#
# The Cartesian core above blows up at the poles when explicit lateral viscosity
# is added (the viscous CFL dt < dx^2/(4 A_h) collapses as dx -> 0 toward the
# poles), and a giant Rayleigh drag is the only thing that holds the spurious
# coarse-grid geostrophic transport down -- which also crushes the real
# overturning. This version fixes both:
#   * proper SPHERICAL metric (cos(phi) factors) in the pressure-gradient force;
#   * IMPLICIT harmonic lateral viscosity solved with the existing CG elliptic
#     machinery -> unconditionally stable for any A_h/dt (no polar CFL, no NaN);
#   * NO-SLIP boundaries: the viscous Laplacian sees land/edges as zero, so a
#     frictional (Munk) western boundary layer forms instead of a free-slip wall.
# This is the building block the realistic prognostic AMOC needs; it is still
# STANDALONE (not wired into veros_driver/main) until the barotropic-baroclinic
# split (S5b) and the spin-up tuning (S5c).
# ===========================================================================
from chronos_esm.ocean.solver import jacobi_preconditioner, solve_cg  # noqa: E402

OMEGA = 7.292e-5  # Earth rotation rate [1/s]
COS_MIN = 0.087  # polar cos(phi) floor (~85 deg); matches solver._sphere_* metric


def _face_conductances(lat, dlon, dlat, a, cos_min=COS_MIN):
    """Unit-coefficient spherical Laplacian face conductances + cell area for a
    centred lat-lon grid, returned as (1, ny, 1) arrays for 3-D (nz, ny, nx)
    broadcasting. Mirrors solver._sphere_conductances with coef == 1."""
    cosc = jnp.maximum(jnp.cos(lat), cos_min)
    lat_n = 0.5 * (lat + jnp.concatenate([lat[1:], lat[-1:]]))
    lat_s = 0.5 * (lat + jnp.concatenate([lat[:1], lat[:-1]]))
    cosn = jnp.maximum(jnp.cos(lat_n), cos_min)
    coss = jnp.maximum(jnp.cos(lat_s), cos_min)
    cE = dlat / (cosc * dlon)
    cW = cE
    cN = cosn * dlon / dlat
    cS = coss * dlon / dlat
    area = (a**2) * cosc * dlon * dlat

    def col(z):
        return z[None, :, None]

    return col(cE), col(cW), col(cN), col(cS), col(area)


def _lap3d(field, cE, cW, cN, cS):
    """Area-weighted spherical operator sum_faces C(field - field_nbr) per level
    (= -area * div(grad), symmetric SPD). No-slip: land/edge neighbours are zero.
    field: (nz, ny, nx); cE.. : (1, ny, 1)."""
    pe = jnp.roll(field, -1, axis=2)  # periodic in longitude
    pw = jnp.roll(field, 1, axis=2)
    z = jnp.zeros_like(field[:, :1, :])
    pn = jnp.concatenate([field[:, 1:, :], z], axis=1)  # no-slip north edge
    ps = jnp.concatenate([z, field[:, :-1, :]], axis=1)  # no-slip south edge
    return cE * (field - pe) + cW * (field - pw) + cN * (field - pn) + cS * (field - ps)


def implicit_viscosity_sphere(
    field, alpha, lat, dlon, dlat, a, mask, max_iter=200, tol=1e-7, cos_min=COS_MIN
):
    """Solve the spherical Helmholtz system (I - alpha * lap) x = field with no-slip
    (x = 0 on land), per level, via preconditioned CG. ``alpha = dt * A_h``.
    Unconditionally stable for any alpha (implicit) -> no viscous CFL / polar blow-up.
    Differentiable. field: (nz, ny, nx); mask: (nz, ny, nx) 1=ocean."""
    nz, ny, nx = field.shape
    cE, cW, cN, cS, area = _face_conductances(lat, dlon, dlat, a, cos_min)
    m3 = mask

    def op(xflat):
        x = xflat.reshape(nz, ny, nx)
        mm = area * x + alpha * _lap3d(x, cE, cW, cN, cS)  # area*(I - alpha*lap)
        return (mm * m3 + x * (1.0 - m3)).flatten()  # identity on land

    diag = (area + alpha * (cE + cW + cN + cS)) * m3 + (1.0 - m3)
    precond = jacobi_preconditioner(jnp.broadcast_to(diag, field.shape).flatten())
    b = (area * field * m3).flatten()
    x, _ = solve_cg(
        op, b, field.flatten(), max_iter=max_iter, tol=tol, preconditioner=precond
    )
    return x.reshape(nz, ny, nx) * m3


def pgf_sphere(p, lat, dlon, dlat, a, rho0=RHO0, mask=None, cos_min=COS_MIN):
    """Spherical horizontal pressure-gradient acceleration from pressure p (nz,ny,nx):
        a_lon = -(1/(rho0 a cos phi)) dp/dlon,  a_lat = -(1/(rho0 a)) dp/dphi.
    Periodic in longitude, one-sided at the lat edges. lat: (ny,) [rad]."""
    cosc = jnp.maximum(jnp.cos(lat), cos_min)[None, :, None]
    dpdl = (jnp.roll(p, -1, axis=2) - jnp.roll(p, 1, axis=2)) / (2.0 * dlon)
    pn = jnp.concatenate([p[:, 1:, :], p[:, -1:, :]], axis=1)
    ps = jnp.concatenate([p[:, :1, :], p[:, :-1, :]], axis=1)
    dpdphi = (pn - ps) / (2.0 * dlat)
    ax = -dpdl / (rho0 * a * cosc)
    ay = -dpdphi / (rho0 * a)
    if mask is not None:
        ax, ay = ax * mask, ay * mask
    return ax, ay


def step_momentum_sphere(
    u,
    v,
    rho,
    *,
    lat,
    dlon,
    dlat,
    a,
    dz,
    dt,
    A_h,
    mask,
    omega=OMEGA,
    r=0.0,
    rho0=RHO0,
    visc_iter=200,
    visc_tol=1e-7,
):
    """One step of the spherical baroclinic momentum core: hydrostatic PGF (from rho)
    -> semi-implicit Coriolis -> implicit harmonic viscosity (no-slip). ``A_h`` is the
    lateral eddy viscosity [m^2/s]; ``r`` an optional weak linear drag [1/s]."""
    f = (2.0 * omega * jnp.sin(lat))[None, :, None]
    p = hydrostatic_pressure(rho, dz)
    ax, ay = pgf_sphere(p, lat, dlon, dlat, a, rho0, mask)
    u1, v1 = coriolis_semi_implicit(u, v, ax - r * u, ay - r * v, f, dt)
    if A_h > 0.0:
        alpha = dt * A_h
        u1 = implicit_viscosity_sphere(
            u1, alpha, lat, dlon, dlat, a, mask, visc_iter, visc_tol
        )
        v1 = implicit_viscosity_sphere(
            v1, alpha, lat, dlon, dlat, a, mask, visc_iter, visc_tol
        )
    return u1 * mask, v1 * mask


def spin_up_sphere(
    rho,
    *,
    lat,
    dlon,
    dlat,
    a,
    dz,
    dt,
    A_h,
    r,
    n_steps,
    mask,
    omega=OMEGA,
    rho0=RHO0,
    visc_iter=200,
    visc_tol=1e-7,
):
    """Integrate the spherical baroclinic momentum from rest under a fixed density
    field to (near) steady state. Differentiable (lax.scan). Returns (u, v)."""
    u = jnp.zeros_like(rho)
    v = jnp.zeros_like(rho)

    def body(carry, _):
        u, v = carry
        u, v = step_momentum_sphere(
            u,
            v,
            rho,
            lat=lat,
            dlon=dlon,
            dlat=dlat,
            a=a,
            dz=dz,
            dt=dt,
            A_h=A_h,
            mask=mask,
            omega=omega,
            r=r,
            rho0=rho0,
            visc_iter=visc_iter,
            visc_tol=visc_tol,
        )
        return (u, v), None

    (u, v), _ = jax.lax.scan(body, (u, v), None, length=n_steps)
    return u, v


__all__ = [
    "hydrostatic_pressure",
    "pressure_gradient_accel",
    "coriolis_semi_implicit",
    "step_momentum",
    "spin_up",
    "pgf_sphere",
    "implicit_viscosity_sphere",
    "step_momentum_sphere",
    "spin_up_sphere",
    "G",
    "RHO0",
    "OMEGA",
]
