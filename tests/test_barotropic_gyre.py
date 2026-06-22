"""P3 / S2 first increment: validate the variable-coefficient elliptic solver and the
prognostic barotropic-vorticity prototype against analytic balances. Pure new numerics --
nothing in the live model (veros_driver/main) is touched, so this cannot regress the
running ocean.

Checks:
  1. Manufactured solution: solve_elliptic_varcoef inverts div(coef*grad) exactly
     (constant AND variable coef) -- the round trip psi -> zeta -> psi.
  2. Constant-coef agreement with the existing solve_poisson_2d.
  3. Stommel gyre spin-up: Sverdrup interior balance (beta*v = curl(tau)/(rho*H)) and
     western intensification (the meridional jet sits on the western boundary).
  4. Differentiability: jax.grad of the gyre strength wrt the wind stress is finite/nonzero.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from chronos_esm.ocean import barotropic as bt  # noqa: E402
from chronos_esm.ocean.solver import (  # noqa: E402
    apply_elliptic_varcoef,
    apply_elliptic_varcoef_sphere,
    solve_elliptic_varcoef,
    solve_elliptic_varcoef_sphere,
    solve_poisson_2d,
)

A_EARTH = 6.371e6


def _basin(ny, nx):
    """Closed basin: land on the 4 edges, ocean interior."""
    m = np.ones((ny, nx))
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0.0
    return jnp.asarray(m)


def test_elliptic_varcoef_manufactured_constant():
    ny, nx = 48, 48
    Lx = Ly = 4.0e6
    dx, dy = Lx / nx, Ly / ny
    mask = _basin(ny, nx)
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys)
    psi0 = jnp.asarray(np.sin(2 * np.pi * X / Lx) * np.sin(np.pi * Y / Ly)) * mask
    coef = jnp.ones((ny, nx))
    zeta = apply_elliptic_varcoef(psi0, coef, dx, dy, mask)
    psi, info = solve_elliptic_varcoef(
        coef, zeta, dx, dy, mask=mask, max_iter=1500, tol=1e-10
    )
    err = jnp.linalg.norm((psi - psi0) * mask) / jnp.linalg.norm(psi0 * mask)
    assert float(err) < 1e-3, f"constant-coef round trip err={float(err):.2e}"


def test_elliptic_varcoef_manufactured_variable():
    ny, nx = 48, 48
    Lx = Ly = 4.0e6
    dx, dy = Lx / nx, Ly / ny
    mask = _basin(ny, nx)
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys)
    psi0 = jnp.asarray(np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)) * mask
    # smooth, strictly positive variable coefficient (e.g. 1/H over varying depth)
    coef = jnp.asarray(1.0 + 0.5 * np.sin(np.pi * X / Lx) * np.cos(np.pi * Y / Ly))
    zeta = apply_elliptic_varcoef(psi0, coef, dx, dy, mask)
    psi, info = solve_elliptic_varcoef(
        coef, zeta, dx, dy, mask=mask, max_iter=2000, tol=1e-10
    )
    err = jnp.linalg.norm((psi - psi0) * mask) / jnp.linalg.norm(psi0 * mask)
    assert float(err) < 2e-3, f"variable-coef round trip err={float(err):.2e}"


def test_elliptic_varcoef_matches_poisson():
    ny, nx = 40, 40
    dx = dy = 1.0e5
    mask = _basin(ny, nx)
    rng = np.random.default_rng(0)
    rhs = jnp.asarray(rng.standard_normal((ny, nx))) * mask
    psi_v, _ = solve_elliptic_varcoef(
        jnp.ones((ny, nx)), rhs, dx, dy, mask=mask, max_iter=2000, tol=1e-10
    )
    # solve_poisson_2d solves lap(psi)=rhs with the same land-identity convention
    psi_p, _ = solve_poisson_2d(rhs, dx, dy, max_iter=2000, tol=1e-10, mask=mask)
    diff = jnp.linalg.norm((psi_v - psi_p) * mask) / (
        jnp.linalg.norm(psi_p * mask) + 1e-30
    )
    assert float(diff) < 1e-3, f"varcoef vs poisson diff={float(diff):.2e}"


# ---- Stommel gyre ---------------------------------------------------------
def _gyre_forcing(ny, nx, Lx, Ly, dx, dy, tau0, rho0, H):
    ys = jnp.asarray((np.arange(ny) + 0.5) * dy)
    taux = (-tau0 * jnp.cos(np.pi * ys / Ly))[:, None] * jnp.ones((ny, nx))
    tauy = jnp.zeros((ny, nx))
    curl = bt.wind_stress_curl(taux, tauy, dx, dy)
    return curl / (rho0 * H), taux


def _run_gyre(tau0, ny=48, nx=48):
    Lx = Ly = 4.0e6
    dx, dy = Lx / nx, Ly / ny
    mask = _basin(ny, nx)
    beta, rho0, H = 2.0e-11, 1025.0, 4000.0
    r = 5.0e-6  # delta = r/beta ~ 250 km ~ 3 dx (resolved WBC)
    F, taux = _gyre_forcing(ny, nx, Lx, Ly, dx, dy, tau0, rho0, H)
    F = F * mask
    psi, zeta = bt.spin_up_gyre(
        F,
        dx=dx,
        dy=dy,
        dt=1000.0,
        beta=beta,
        r=r,
        mask=mask,
        coef=jnp.ones((ny, nx)),
        n_steps=900,
        max_iter=120,
    )
    return psi, F, mask, beta, dx, dy


def test_stommel_gyre_sverdrup_and_western_intensification():
    psi, F, mask, beta, dx, dy = _run_gyre(0.1)
    psi = np.asarray(psi)
    assert np.all(np.isfinite(psi)), "gyre psi not finite"

    # western intensification: the meridional jet |v|=|dpsi/dx| peaks near the west wall.
    v = np.asarray(bt.velocities_from_psi(jnp.asarray(psi), dx, dy, mask)[1])
    ny, nx = psi.shape
    midrows = slice(ny // 3, 2 * ny // 3)
    vmag_by_col = np.abs(v[midrows, :]).mean(axis=0)
    jet_col = int(np.argmax(vmag_by_col))
    assert jet_col < nx // 3, f"jet at col {jet_col}/{nx} -- not western-intensified"

    # Sverdrup interior: dpsi/dx ~ F/beta away from the western boundary layer.
    psix = np.asarray(bt.ddx_centered(jnp.asarray(psi), dx))
    Fn = np.asarray(F)
    interior = (slice(ny // 3, 2 * ny // 3), slice(nx // 2, nx - 3))
    pred = Fn[interior] / beta
    got = psix[interior]
    rel = np.abs(got - pred).mean() / (np.abs(pred).mean() + 1e-30)
    assert rel < 0.35, f"interior Sverdrup balance rel.err={rel:.2f}"


def test_gyre_differentiable():
    def strength(tau0):
        psi, _, _, _, _, _ = _run_gyre(tau0, ny=32, nx=32)
        return jnp.max(jnp.abs(psi))

    g = jax.grad(strength)(0.1)
    assert np.isfinite(float(g)) and abs(float(g)) > 0, f"d(gyre)/d(tau0)={float(g)}"


# ---- spherical elliptic operator (P3/S2 second half) ----------------------
def _sphere_band(ny, nx, lat0_deg=20.0, lat1_deg=70.0):
    """Periodic-in-lon lat band; land walls on the north/south edges."""
    lat = np.deg2rad(np.linspace(lat0_deg, lat1_deg, ny))
    dlat = float(lat[1] - lat[0])
    dlon = 2.0 * np.pi / nx
    m = np.ones((ny, nx))
    m[0, :] = m[-1, :] = 0.0  # Dirichlet psi=0 at band edges
    return jnp.asarray(lat), dlat, dlon, jnp.asarray(m)


def test_sphere_elliptic_manufactured_variable():
    ny, nx = 48, 64
    lat, dlat, dlon, mask = _sphere_band(ny, nx)
    lon = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    LON, LAT = np.meshgrid(lon, np.asarray(lat))
    lat0, lat1 = float(lat[0]), float(lat[-1])
    env = np.sin(np.pi * (np.asarray(lat)[:, None] - lat0) / (lat1 - lat0))
    psi0 = jnp.asarray(np.sin(2 * LON) * env) * mask  # periodic in lon, 0 at edges
    coef = jnp.asarray(1.0 + 0.4 * np.sin(LON) * np.cos(LAT))  # variable 1/H-like, >0
    rhs = apply_elliptic_varcoef_sphere(psi0, coef, lat, dlon, dlat, A_EARTH, mask)
    psi, info = solve_elliptic_varcoef_sphere(
        coef, rhs, lat, dlon, dlat, A_EARTH, mask=mask, max_iter=2500, tol=1e-9
    )
    # NB: the residual stalls at a float64 roundoff floor in this anisotropic system
    # (cE~1/cos, cN~cos), but the SOLUTION error is the meaningful metric and is tiny.
    err = jnp.linalg.norm((psi - psi0) * mask) / jnp.linalg.norm(psi0 * mask)
    assert float(err) < 1e-4, f"spherical var-coef round trip err={float(err):.2e}"


def test_sphere_elliptic_differentiable():
    ny, nx = 32, 40
    lat, dlat, dlon, mask = _sphere_band(ny, nx)
    rng = np.random.default_rng(1)
    rhs = jnp.asarray(rng.standard_normal((ny, nx))) * mask

    def amp(scale):
        coef = scale * jnp.ones((ny, nx))  # coef ~ 1/H; vary uniformly
        psi, _ = solve_elliptic_varcoef_sphere(
            coef, rhs, lat, dlon, dlat, A_EARTH, mask=mask, max_iter=3000, tol=1e-10
        )
        return jnp.max(jnp.abs(psi))

    g = jax.grad(amp)(1.0)
    assert np.isfinite(float(g)) and abs(float(g)) > 0, f"d|psi|/d(coef)={float(g)}"


OMEGA = 7.292e-5


def _run_gyre_sphere(tau0, ny=44, nx=40):
    """Mid-latitude sector basin (15-60N, 0-60E), closed by land walls."""
    lat = np.deg2rad(np.linspace(15.0, 60.0, ny))
    dlat = float(lat[1] - lat[0])
    lon = np.deg2rad(np.linspace(0.0, 60.0, nx))
    dlon = float(lon[1] - lon[0])
    m = np.ones((ny, nx))
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0.0
    mask = jnp.asarray(m)
    lat_j = jnp.asarray(lat)
    rho0, H, r = 1025.0, 4000.0, 6.0e-6
    lat0, lat1 = lat[0], lat[-1]
    taux = (-tau0 * jnp.cos(np.pi * (lat_j - lat0) / (lat1 - lat0)))[
        :, None
    ] * jnp.ones((ny, nx))
    tauy = jnp.zeros((ny, nx))
    curl = bt.wind_stress_curl_sphere(taux, tauy, lat_j, dlon, dlat, A_EARTH)
    F = (curl / (rho0 * H)) * mask
    psi, zeta = bt.spin_up_gyre_sphere(
        F,
        lat=lat_j,
        dlon=dlon,
        dlat=dlat,
        a=A_EARTH,
        omega=OMEGA,
        dt=1500.0,
        r=r,
        mask=mask,
        coef=jnp.ones((ny, nx)),
        n_steps=800,
        max_iter=150,
    )
    return psi, F, mask, lat_j, dlon, dlat


def test_stommel_gyre_sphere():
    psi, F, mask, lat_j, dlon, dlat = _run_gyre_sphere(0.1)
    psi = np.asarray(psi)
    assert np.all(np.isfinite(psi)), "spherical gyre psi not finite"
    ny, nx = psi.shape

    # western intensification: meridional jet |v| peaks near the west wall
    _, v = bt.velocities_sphere(jnp.asarray(psi), lat_j, dlon, dlat, A_EARTH, mask)
    v = np.asarray(v)
    midrows = slice(ny // 3, 2 * ny // 3)
    jet_col = int(np.argmax(np.abs(v[midrows, :]).mean(axis=0)))
    assert jet_col < nx // 3, f"jet at col {jet_col}/{nx} -- not western-intensified"

    # Sverdrup interior: v ~ F/beta away from the western boundary layer
    beta = (2.0 * OMEGA * np.cos(np.asarray(lat_j)) / A_EARTH)[:, None]
    pred = np.asarray(F) / beta
    interior = (slice(ny // 3, 2 * ny // 3), slice(nx // 2, nx - 3))
    rel = np.abs(v[interior] - pred[interior]).mean() / (
        np.abs(pred[interior]).mean() + 1e-30
    )
    assert rel < 0.45, f"spherical Sverdrup balance rel.err={rel:.2f}"


def test_gyre_sphere_differentiable():
    def strength(tau0):
        psi, _, _, _, _, _ = _run_gyre_sphere(tau0, ny=28, nx=28)
        return jnp.max(jnp.abs(psi))

    g = jax.grad(strength)(0.1)
    assert (
        np.isfinite(float(g)) and abs(float(g)) > 0
    ), f"d(gyre_sphere)/d(tau0)={float(g)}"


if __name__ == "__main__":
    test_elliptic_varcoef_manufactured_constant()
    test_elliptic_varcoef_manufactured_variable()
    test_elliptic_varcoef_matches_poisson()
    test_stommel_gyre_sverdrup_and_western_intensification()
    test_gyre_differentiable()
    print("all barotropic-gyre tests passed")
