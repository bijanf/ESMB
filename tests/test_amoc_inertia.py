"""The two fixes that solve the COUPLED-model AMOC noise (see
experiments/diagnose_coupled_amoc.py): (1) the THC overturning amplitude is relaxed with
temporal inertia at the coupling level; (2) compute_amoc removes the barotropic per WET
COLUMN so the wind-driven barotropic gyre cannot leak into the overturning over variable
bathymetry. These are fast, network-free unit checks of the underlying pieces.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from chronos_esm.config import OCEAN_DZ, OCEAN_GRID  # noqa: E402
from chronos_esm.ocean import overturning  # noqa: E402
from chronos_esm.ocean.diagnostics import (  # noqa: E402
    compute_amoc,
    create_atlantic_mask,
)
from chronos_esm.ocean.veros_driver import (  # noqa: E402
    equation_of_state,
    init_ocean_state,
)

NZ, NY, NX = OCEAN_GRID.nz, OCEAN_GRID.nlat, OCEAN_GRID.nlon
DZ = jnp.asarray(OCEAN_DZ)
LAT = np.linspace(-90, 90, NY)


def test_thc_target_amplitude_positive_and_density_responsive():
    """The instantaneous THC target amp is >=0 and increases when the subpolar ocean
    is made denser (saltier) -- the density pathway the inertia must preserve."""
    maskC = jnp.ones((NZ, NY, NX))
    salt = jnp.ones((NZ, NY, NX)) * 35.0
    base_t = 285.0 - 12.0 * jnp.asarray(np.abs(np.sin(np.deg2rad(LAT))))[None, :, None]
    temp = jnp.broadcast_to(base_t, (NZ, NY, NX))

    def amp_for(dssub):
        sp = ((LAT >= 45) & (LAT <= 65)).astype(np.float64)[None, :, None]
        s = salt + jnp.asarray(np.broadcast_to(sp, (NZ, NY, NX))) * dssub
        rho = equation_of_state(temp, s)
        return float(overturning.thc_target_amplitude(rho, s, DZ, maskC, k_vel=1.0e-4))

    a0 = amp_for(0.0)
    a_salty = amp_for(0.5)  # saltier/denser subpolar
    assert a0 >= 0.0 and np.isfinite(a0)
    assert a_salty > a0, "denser subpolar must strengthen the THC amplitude"


def test_amp_override_bypasses_softplus():
    """Passing amp_override sets the overturning amplitude directly (the inertial path)."""
    maskC = jnp.ones((NZ, NY, NX))
    rho = equation_of_state(
        jnp.ones((NZ, NY, NX)) * 283.0, jnp.ones((NZ, NY, NX)) * 35.0
    )
    salt = jnp.ones((NZ, NY, NX)) * 35.0
    v_over, _, amp = overturning.thc_overturning_velocity(
        rho, salt, DZ, maskC, k_vel=1.0e-4, amp_override=jnp.asarray(7e-4)
    )
    assert abs(float(amp) - 7e-4) < 1e-12
    assert bool(jnp.isfinite(v_over).all())


def _variable_bathymetry_mask():
    """3D wet mask whose lower layers are land in part of the basin (variable depth)."""
    m = np.ones((NZ, NY, NX))
    m[NZ // 2 :, :, : NX // 2] = 0.0  # western half is shallow (half-depth)
    return jnp.asarray(m)


def test_per_column_removal_kills_barotropic_gyre_leak():
    """A PURE depth-uniform (barotropic) gyre has ZERO overturning. Per-column wet-depth
    removal recovers ~0; the coarse per-latitude/full-depth removal LEAKS it over variable
    bathymetry (the coupled-AMOC noise source)."""
    mask3d = _variable_bathymetry_mask()
    omask2d = mask3d[0] > 0.5
    atl = create_atlantic_mask(NY, NX)
    # depth-uniform meridional velocity (a barotropic gyre), wet cells only
    vbt = np.zeros((NZ, NY, NX))
    vy = np.sin(np.deg2rad(LAT) * 3.0)[None, :, None]  # some lateral structure
    vbt[:] = vy
    v = jnp.asarray(vbt) * mask3d
    st = init_ocean_state(NZ, NY, NX)._replace(v=v)

    per_lat = float(
        compute_amoc(st, atlantic_mask=atl, dz=DZ, ocean_mask=omask2d)["upper_cell_26N"]
    )
    per_col = float(
        compute_amoc(
            st, atlantic_mask=atl, dz=DZ, ocean_mask=omask2d, ocean_mask_3d=mask3d
        )["upper_cell_26N"]
    )
    assert abs(per_col) < 1e-6, f"per-column should give ~0 overturning, got {per_col}"
    assert abs(per_lat) > 10 * max(
        abs(per_col), 1e-9
    ), f"per-latitude should LEAK the barotropic gyre (got {per_lat} vs {per_col})"


if __name__ == "__main__":
    test_thc_target_amplitude_positive_and_density_responsive()
    test_amp_override_bypasses_softplus()
    test_per_column_removal_kills_barotropic_gyre_leak()
    print("all AMOC inertia / per-column tests passed")
