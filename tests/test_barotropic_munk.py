"""S5b: the MUNK barotropic streamfunction (barotropic.spin_up_munk_gyre_sphere).

The Stommel cores close the western boundary layer with a Rayleigh drag (Stommel
width r/beta); the Munk core closes it with LATERAL VISCOSITY (Munk width
(A_h/beta)^(1/3)). Network-free idealized mid-latitude basin checks:

  1. a Munk gyre forms (finite, western-intensified, Sverdrup interior);
  2. the western boundary-layer width scales as (A_h/beta)^(1/3) -- the analytic
     Munk signature (doubling-ish for an 8x viscosity, NOT 8x);
  3. it is differentiable.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from chronos_esm.ocean import barotropic as bt  # noqa: E402

A_EARTH = 6.371e6
OMEGA = 7.292e-5


def _munk_basin(ny=44, nx=40):
    """Closed mid-latitude sector basin (15-60N, 0-40E) with a single-gyre wind curl."""
    lat = np.deg2rad(np.linspace(15.0, 60.0, ny))
    dlat = float(lat[1] - lat[0])
    lon = np.deg2rad(np.linspace(0.0, 40.0, nx))
    dlon = float(lon[1] - lon[0])
    m = np.ones((ny, nx))
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = 0.0
    mask = jnp.asarray(m)
    lat_j = jnp.asarray(lat)
    tau0, rho0, H = 0.1, 1025.0, 4000.0
    taux = (-tau0 * jnp.cos(np.pi * (lat_j - lat[0]) / (lat[-1] - lat[0])))[
        :, None
    ] * jnp.ones((ny, nx))
    curl = bt.wind_stress_curl_sphere(
        taux, jnp.zeros((ny, nx)), lat_j, dlon, dlat, A_EARTH
    )
    F = (curl / (rho0 * H)) * mask
    return F, mask, lat_j, dlon, dlat


def _run_munk(A_h, ny=44, nx=40, n_steps=1200):
    F, mask, lat_j, dlon, dlat = _munk_basin(ny, nx)
    psi, zeta = bt.spin_up_munk_gyre_sphere(
        F,
        lat=lat_j,
        dlon=dlon,
        dlat=dlat,
        a=A_EARTH,
        omega=OMEGA,
        dt=1500.0,
        r=0.0,
        A_h=A_h,
        mask=mask,
        coef=jnp.ones((ny, nx)),
        n_steps=n_steps,
        max_iter=200,
        visc_iter=120,
    )
    return psi, F, mask, lat_j, dlon, dlat


def _west_width(psi, lat_j, dlon, dlat, mask):
    """Western-boundary-current e-folding width [m]: distance east of the jet peak
    where the mid-latitude |v| falls to 1/e of its peak. This measures the boundary-
    layer decay scale (~ Munk delta_M) and is insensitive to the broad interior return
    flow (which a centroid over the whole western half is not)."""
    _, v = bt.velocities_sphere(psi, lat_j, dlon, dlat, A_EARTH, mask)
    v = np.abs(np.asarray(v))
    ny, nx = v.shape
    vcol = v[ny // 3 : 2 * ny // 3, :].mean(axis=0)  # (nx,) mean |v| by column
    phi_mid = float(np.asarray(lat_j)[ny // 2])
    dxc = A_EARTH * np.cos(phi_mid) * dlon
    peak = int(np.argmax(vcol))
    pv = vcol[peak]
    thr = pv / np.e
    for i in range(peak + 1, nx):
        if vcol[i] < thr:  # first crossing east of the peak (linearly interpolated)
            frac = (vcol[i - 1] - thr) / (vcol[i - 1] - vcol[i] + 1e-30)
            return (i - 1 + frac - peak) * dxc
    return (nx - 1 - peak) * dxc


def test_munk_gyre_forms_and_western_intensified():
    psi, F, mask, lat_j, dlon, dlat = _run_munk(6.0e5)
    psi_np = np.asarray(psi)
    assert np.all(np.isfinite(psi_np)), "Munk gyre psi not finite"
    assert float(jnp.max(jnp.abs(psi))) > 0.0
    # western intensification: meridional jet peaks in the western third
    _, v = bt.velocities_sphere(psi, lat_j, dlon, dlat, A_EARTH, mask)
    v = np.asarray(v)
    ny, nx = psi_np.shape
    jet_col = int(np.argmax(np.abs(v[ny // 3 : 2 * ny // 3, :]).mean(axis=0)))
    assert jet_col < nx // 3, f"jet at col {jet_col}/{nx} -- not western-intensified"


def test_munk_boundary_layer_scaling():
    """8x viscosity -> Munk width grows by ~8^(1/3)=2.0 (NOT 8). The discriminating
    test that the boundary layer is Munk (A_h), not Stommel (drag)."""
    A_lo, A_hi = 3.0e5, 2.4e6  # ratio 8
    p_lo, _, mask, lat_j, dlon, dlat = _run_munk(A_lo)
    p_hi, _, _, _, _, _ = _run_munk(A_hi)
    w_lo = _west_width(p_lo, lat_j, dlon, dlat, mask)
    w_hi = _west_width(p_hi, lat_j, dlon, dlat, mask)
    ratio = w_hi / w_lo
    theory = (A_hi / A_lo) ** (1.0 / 3.0)  # = 2.0
    # the boundary current widens with A_h^(1/3): clearly > 1.4, near 2, far below 8
    assert 1.4 < ratio < 3.2, f"Munk width ratio {ratio:.2f} (theory {theory:.2f})"
    assert ratio < 4.0, "width scaling looks like Stommel/linear, not Munk^(1/3)"


def test_munk_gyre_differentiable():
    def strength(tau_scale):
        F, mask, lat_j, dlon, dlat = _munk_basin(ny=30, nx=28)
        psi, _ = bt.spin_up_munk_gyre_sphere(
            F * tau_scale,
            lat=lat_j,
            dlon=dlon,
            dlat=dlat,
            a=A_EARTH,
            omega=OMEGA,
            dt=1500.0,
            r=0.0,
            A_h=6.0e5,
            mask=mask,
            coef=jnp.ones_like(F),
            n_steps=400,
            max_iter=150,
            visc_iter=80,
        )
        return jnp.max(jnp.abs(psi))

    g = jax.grad(strength)(1.0)
    assert np.isfinite(float(g)) and abs(float(g)) > 0, f"d(gyre)/d(tau)={float(g)}"


if __name__ == "__main__":
    test_munk_gyre_forms_and_western_intensified()
    test_munk_boundary_layer_scaling()
    test_munk_gyre_differentiable()
    print("all S5b Munk-gyre tests passed")
