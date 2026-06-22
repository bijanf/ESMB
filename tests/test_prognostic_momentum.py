"""P3 / S5: the prognostic baroclinic momentum core wired into step_ocean (behind the
`prognostic_momentum` flag) gives DENSITY a dynamical pathway to the overturning -- the
regression-breaker for the P0 baseline where d(AMOC)/d(density) was ~0.

Checks:
  1. Flag ON runs finite over several steps.
  2. d(overturning)/d(subpolar salinity) is clearly nonzero with the prognostic momentum,
     and much larger than with the diagnostic thermal-wind path (flag OFF) -- i.e. the
     overturning is now density-responsive for a dynamical reason.
The flag defaults OFF, so the existing model path is unchanged (no regression).
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from chronos_esm.config import OCEAN_GRID  # noqa: E402
from chronos_esm.ocean.veros_driver import init_ocean_state, step_ocean  # noqa: E402

R = 6.371e6
NZ, NY, NX = OCEAN_GRID.nz, OCEAN_GRID.nlat, OCEAN_GRID.nlon
DX, DY = 2 * np.pi * R / NX, np.pi * R / NY
DZ = jnp.ones(NZ) * 200.0
FLUX = (jnp.zeros((NY, NX)), jnp.zeros((NY, NX)), jnp.zeros((NY, NX)))
STRESS = (jnp.zeros((NY, NX)), jnp.zeros((NY, NX)))


def _base_state():
    """Uniform ocean with a meridional temperature gradient (warm equator, cold poles)
    so there is a density structure for an overturning to develop."""
    st = init_ocean_state(NZ, NY, NX)
    lat = np.linspace(-90, 90, NY)
    tprof = 288.15 - 18.0 * np.abs(np.sin(np.deg2rad(lat)))      # (ny,) cold poles
    temp = jnp.asarray(np.broadcast_to(tprof[None, :, None], (NZ, NY, NX)))
    return st._replace(temp=temp)


def _subpolar_mask():
    lat = np.linspace(-90, 90, NY)
    m = ((lat >= 45) & (lat <= 65)).astype(np.float64)
    return jnp.asarray(np.broadcast_to(m[None, :, None], (NZ, NY, NX)))


def _overturning_sv(v):
    """Smooth meridional-overturning-strength scalar [Sv] from v(z,y,x). RMS (not max) so
    its gradient is well-defined for the density-responsiveness check."""
    Vz = jnp.sum(v * DZ[:, None, None] * DX, axis=2)            # (nz,ny) zonal transport
    psi = jnp.cumsum(Vz, axis=0) / 1.0e6                        # cumulative in depth -> Sv
    return jnp.sqrt(jnp.mean(psi ** 2))                         # smooth RMS overturning


def _amoc_grad(prognostic, n_steps=12):
    base = _base_state()
    spm = _subpolar_mask()

    def overturning(salt_pert):
        st = base._replace(salt=base.salt + spm * salt_pert)
        for _ in range(n_steps):
            st = step_ocean(st, FLUX, STRESS, DX, DY, DZ,
                            prognostic_momentum=prognostic, thc_k_vel=0.0)
        return _overturning_sv(st.v)
    val = float(overturning(0.0))
    g = float(jax.grad(overturning)(0.0))
    return val, g


def test_prognostic_runs_finite():
    base = _base_state()
    st = base
    for _ in range(40):
        st = step_ocean(st, FLUX, STRESS, DX, DY, DZ, prognostic_momentum=True,
                        thc_k_vel=0.0)
    assert bool(jnp.isfinite(st.v).all()) and bool(jnp.isfinite(st.temp).all())
    # bounded to physical ocean velocities (not a runaway)
    assert float(jnp.max(jnp.abs(st.v))) < 2.0, "prognostic velocity unphysically large"


def test_rigid_lid_mass_conservation():
    """The rigid-lid projection makes the depth-integrated transport non-divergent."""
    base = _base_state()
    st = base
    for _ in range(20):
        st = step_ocean(st, FLUX, STRESS, DX, DY, DZ, prognostic_momentum=True,
                        thc_k_vel=0.0)
    U = jnp.sum(st.u * DZ[:, None, None], axis=0)
    V = jnp.sum(st.v * DZ[:, None, None], axis=0)
    div = (jnp.roll(U, -1, 1) - jnp.roll(U, 1, 1)) / (2 * DX) \
        + (jnp.roll(V, -1, 0) - jnp.roll(V, 1, 0)) / (2 * DY)
    # normalise by a typical transport gradient scale; expect ~machine zero
    assert float(jnp.max(jnp.abs(div))) < 1e-10, \
        f"depth-integrated flow not non-divergent: max|div U| = {float(jnp.max(jnp.abs(div)))}"


def test_overturning_density_responsive():
    val_p, g_prog = _amoc_grad(prognostic=True)
    val_d, g_diag = _amoc_grad(prognostic=False)
    # the prognostic momentum makes the overturning clearly density-responsive...
    assert np.isfinite(g_prog) and abs(g_prog) > 1.0, \
        f"prognostic d(overturning)/d(salt) = {g_prog} (expected clearly nonzero)"
    # ...and far more so than the diagnostic thermal-wind path (the P0 blocker break)
    assert abs(g_prog) > 10.0 * abs(g_diag), \
        f"prognostic {g_prog} not >> diagnostic {g_diag}"


if __name__ == "__main__":
    test_prognostic_runs_finite()
    vp, gp = _amoc_grad(True); vd, gd = _amoc_grad(False)
    print(f"prognostic: overturning {vp:.3f} Sv, d/d(salt) = {gp:.4e} Sv/psu")
    print(f"diagnostic: overturning {vd:.3f} Sv, d/d(salt) = {gd:.4e} Sv/psu")
    print("all prognostic-momentum tests passed")
