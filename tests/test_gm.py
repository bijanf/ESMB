"""P3: the Gent-McWilliams eddy parameterization (chronos_esm/ocean/gm.py).

Validates the AD-safe GM bolus scheme used by step_ocean(gm_on=True):
  1. flat (horizontally uniform) isopycnals -> zero bolus;
  2. the bolus is depth-integral-zero (Psi*=0 BCs -> carries no net transport);
  3. the slope + surface tapers behave (gentle ~1, steep ->0; ramped to the lid);
  4. AD is finite even through a near-zero-stratification column;
  5. step_ocean(gm_on=True) runs finite and PRESERVES density-responsiveness
     (d(overturning)/d(subpolar salt) != 0 -- GM must not zero the sensitivity).

The quantitative "GM cuts the T31 AMOC barrier by only ~1%" result is in
experiments/diagnose_gm_amoc.py (needs the cached WOA climatology); here we keep to
fast, network-free unit checks on a synthetic stratified ocean.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from chronos_esm.config import OCEAN_DZ, OCEAN_GRID  # noqa: E402
from chronos_esm.ocean import gm  # noqa: E402
from chronos_esm.ocean.veros_driver import (  # noqa: E402
    equation_of_state,
    init_ocean_state,
    step_ocean,
)

NZ, NY, NX = OCEAN_GRID.nz, OCEAN_GRID.nlat, OCEAN_GRID.nlon
R = 6.371e6
DZ = jnp.asarray(OCEAN_DZ)
DY = (np.pi * R) / NY
DX_3D = gm.latitude_dx(NY, NX)
LAT = np.linspace(-90, 90, NY)
DEPTH = np.cumsum(np.asarray(OCEAN_DZ)) - 0.5 * np.asarray(
    OCEAN_DZ
)  # layer centers [m]


def _stratified_temp():
    """Warm-equator / cold-pole SST + a warm-surface/cold-deep thermocline, so the
    density field is STABLY stratified (d rho/dz > 0) with GENTLE isopycnal slopes --
    the regime where GM is active (steep mixed-layer slopes are tapered off)."""
    tsurf = 288.15 - 25.0 * np.sin(np.deg2rad(LAT)) ** 2  # (ny,)
    tz = np.exp(-DEPTH / 800.0)  # 1 at surface -> 0 deep (ny,)
    t = 270.0 + (tsurf[None, :] - 270.0) * tz[:, None]  # (nz, ny)
    return jnp.asarray(np.broadcast_to(t[:, :, None], (NZ, NY, NX)))


def _stratified_state():
    st = init_ocean_state(NZ, NY, NX)
    temp = _stratified_temp()
    return st._replace(temp=temp, rho=equation_of_state(temp, st.salt))


def test_flat_isopycnals_give_zero_bolus():
    """rho depending on z only (horizontally uniform) -> Sx=Sy=0 -> u*=v*=0."""
    rho = jnp.broadcast_to((1025.0 + 0.1 * jnp.arange(NZ))[:, None, None], (NZ, NY, NX))
    u_star, v_star = gm.eddy_induced_velocity(1000.0, rho, DX_3D, DY, DZ)
    assert float(jnp.max(jnp.abs(u_star))) < 1e-12
    assert float(jnp.max(jnp.abs(v_star))) < 1e-12


def test_bolus_is_depth_integral_zero():
    """Psi*=0 at surface & floor => integral_z u* dz = integral_z v* dz = 0 exactly
    (the eddy bolus carries NO net transport, only overturning)."""
    rho = _stratified_state().rho
    u_star, v_star = gm.eddy_induced_velocity(1000.0, rho, DX_3D, DY, DZ)
    dz3 = DZ[:, None, None]
    iu = jnp.sum(u_star * dz3, axis=0)
    iv = jnp.sum(v_star * dz3, axis=0)
    assert float(jnp.max(jnp.abs(iu))) < 1e-10
    assert float(jnp.max(jnp.abs(iv))) < 1e-10


def test_slope_taper_kills_steep_slopes():
    gentle = gm.slope_taper(jnp.array(1e-4), jnp.array(0.0))
    steep = gm.slope_taper(jnp.array(2e-2), jnp.array(0.0))
    assert float(gentle) > 0.95, "gentle slope should be ~untapered"
    assert float(steep) < 0.05, "steep slope should be tapered off"


def test_surface_taper_ramps_to_lid():
    ramp = np.asarray(gm.surface_taper(DZ)).ravel()
    assert ramp[0] < ramp[-1], "taper must increase with depth"
    assert ramp[0] < 0.5, "bolus suppressed near the surface"
    assert ramp[-1] > 0.9, "bolus near full strength at depth"


def test_ad_finite_through_weak_stratification():
    """jax.grad of a bolus scalar w.r.t. rho must be finite even where d rho/dz ~ 0
    (the eps floor + smooth tapers/softclip guarantee no NaNs)."""
    rho0 = _stratified_state().rho
    # inject a near-neutral (zero-stratification) column to exercise the eps floor
    rho0 = rho0.at[:, NY // 2, NX // 2].set(1025.0)

    def scalar(rho):
        u_star, v_star = gm.eddy_induced_velocity(1000.0, rho, DX_3D, DY, DZ)
        return jnp.sum(u_star**2 + v_star**2)

    g = jax.grad(scalar)(rho0)
    assert bool(jnp.isfinite(g).all()), "GM gradient is not finite"


def test_gm_on_runs_finite_and_density_responsive():
    """step_ocean(gm_on=True) runs finite, and the prognostic overturning stays
    density-responsive (GM must not zero d(AMOC)/d(density))."""
    base = _stratified_state()
    flux = (jnp.zeros((NY, NX)),) * 3
    stress = (jnp.zeros((NY, NX)),) * 2
    dx = DX_3D[0]  # (ny,1), as production passes a latitude-dependent dx

    def overturning(salt_pert):
        spm = jnp.asarray(
            np.broadcast_to(
                ((LAT >= 45) & (LAT <= 65)).astype(np.float64)[None, :, None],
                (NZ, NY, NX),
            )
        )
        st = base._replace(salt=base.salt + spm * salt_pert)
        for _ in range(8):
            st = step_ocean(
                st,
                flux,
                stress,
                dx,
                DY,
                DZ,
                prognostic_momentum=True,
                thc_k_vel=0.0,
                gm_on=True,
            )
        vz = jnp.sum(st.v * DZ[:, None, None] * dx, axis=2)
        psi = jnp.cumsum(vz, axis=0) / 1.0e6
        return jnp.sqrt(jnp.mean(psi**2))

    val = float(overturning(0.0))
    g = float(jax.grad(overturning)(0.0))
    assert np.isfinite(val), "overturning not finite"
    assert np.isfinite(g) and abs(g) > 1.0, f"GM zeroed density sensitivity: g={g}"


if __name__ == "__main__":
    test_flat_isopycnals_give_zero_bolus()
    test_bolus_is_depth_integral_zero()
    test_slope_taper_kills_steep_slopes()
    test_surface_taper_ramps_to_lid()
    test_ad_finite_through_weak_stratification()
    test_gm_on_runs_finite_and_density_responsive()
    print("all GM tests passed")
