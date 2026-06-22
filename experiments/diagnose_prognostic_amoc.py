"""Dissect why the prognostic ocean core (P3/S5) equilibrates to AMOC ~ 0 Sv.

Loads a coupled checkpoint (ocean__* flat keys) and reports:
  - global MOC streamfunction (lat-depth) max/min + locations
  - Atlantic AMOC streamfunction (depth-mean removed) max/min + 26.5N profile
  - meridional velocity statistics (is v large but zonally cancelling, or genuinely small?)
  - density structure (N-S surface gradient, top-bottom stratification, N.Atl convection)
  - the DIAGNOSTIC thermal-wind overturning on the SAME density, for comparison
  - the depth-integrated meridional transport per latitude (rigid-lid residual)

Usage: python experiments/diagnose_prognostic_amoc.py <checkpoint.npz> [more.npz ...]
"""
import sys
import os

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm.ocean.veros_driver import OceanState, init_ocean_state  # noqa: E402
from chronos_esm.ocean.diagnostics import (compute_amoc, compute_moc,  # noqa: E402
                                           create_atlantic_mask)
from chronos_esm.config import OCEAN_GRID, OCEAN_DZ, EARTH_RADIUS  # noqa: E402


def load_ocean(path):
    d = np.load(path)
    return OceanState(
        u=jnp.asarray(d["ocean__u"]), v=jnp.asarray(d["ocean__v"]),
        w=jnp.asarray(d["ocean__w"]), temp=jnp.asarray(d["ocean__temp"]),
        salt=jnp.asarray(d["ocean__salt"]), psi=jnp.asarray(d["ocean__psi"]),
        rho=jnp.asarray(d["ocean__rho"]), dic=jnp.asarray(d["ocean__dic"]),
    ), int(d["day"]) if "day" in d else -1


def ocean_mask_from_state(st):
    """Wet cells ~ where temp is not the land fill / finite & nonzero density column."""
    # surface wet where temp differs from a land sentinel; use rho>0 as wet proxy
    wet = np.asarray(st.rho[0] > 0) & np.isfinite(np.asarray(st.temp[0]))
    return jnp.asarray(wet)


def report(path):
    st, day = load_ocean(path)
    nz, ny, nx = st.v.shape
    lat = np.linspace(-90, 90, ny)
    dz = np.asarray(OCEAN_DZ)
    H = dz.sum()
    omask = ocean_mask_from_state(st)
    amask = create_atlantic_mask(ny, nx)

    print(f"\n{'='*72}\n{os.path.basename(path)}   day {day} ({day/365:.2f} yr)   "
          f"grid {nz}x{ny}x{nx}\n{'='*72}")

    # --- velocity magnitudes ---
    u = np.asarray(st.u); v = np.asarray(st.v); w = np.asarray(st.w)
    print(f"|u|max {np.abs(u).max():.3f}  |v|max {np.abs(v).max():.3f}  "
          f"|w|max {np.abs(w).max():.2e} m/s   "
          f"v RMS {np.sqrt(np.mean(v**2)):.4f}")

    # --- global MOC ---
    moc = np.asarray(compute_moc(st))                          # (nz,ny) Sv
    iz, iy = np.unravel_index(np.argmax(np.abs(moc)), moc.shape)
    print(f"global MOC: max {moc.max():+.1f}  min {moc.min():+.1f} Sv  "
          f"(extreme at lat {lat[iy]:.0f}, level {iz})")

    # --- Atlantic AMOC (depth-mean removed, the dashboard metric) ---
    res = compute_amoc(st, ocean_mask=omask)
    psi = np.asarray(res["streamfunction"])                    # (nz,ny)
    i26 = int(np.argmin(np.abs(lat - 26.5)))
    print(f"Atlantic AMOC: upper_26N {float(res['upper_cell_26N']):+.2f}  "
          f"lower_26N {float(res['lower_cell_26N']):+.2f} Sv   "
          f"basin max {psi.max():+.1f}  min {psi.min():+.1f}")
    print(f"  26.5N profile (Sv, top->bot): "
          f"{np.array2string(psi[:, i26], precision=2, max_line_width=200)}")

    # --- meridional transport per latitude (BEFORE depth-mean removal) ---
    # zonal-integrated v over Atlantic, depth-integrated -> net transport per lat
    dx = 2*np.pi*EARTH_RADIUS*np.cos(np.deg2rad(lat))/nx       # (ny,)
    basin = np.asarray(amask) & np.asarray(omask)
    v_atl = np.where(basin[None], v, 0.0)
    v_zonal = np.sum(v_atl * dx[None, :, None], axis=2)        # (nz,ny) m^2/s
    net_transport = np.sum(v_zonal * dz[:, None], axis=0) / 1e6   # (ny,) Sv net
    print(f"  Atlantic NET meridional transport per-lat: "
          f"max |{np.abs(net_transport).max():.2f}| Sv "
          f"(rigid-lid residual; should be ~0 if mass-conserving)")
    # baroclinic content: how much depth structure is there in v_zonal after demean?
    bt = np.sum(v_zonal * dz[:, None], axis=0, keepdims=True) / H
    v_bc = v_zonal - bt
    print(f"  Atlantic baroclinic |v_zonal| RMS {np.sqrt(np.mean(v_bc**2)):.3e} m^2/s "
          f"(0 => no overturning structure to integrate)")

    # --- density / buoyancy structure (the AMOC driver) ---
    rho = np.asarray(st.rho)
    surf = rho[0]
    sgrad = np.where(np.asarray(omask), surf, np.nan)
    # N-S surface density contrast (equator vs subpolar N.Atl)
    eq = np.nanmean(np.where(np.abs(lat) < 10, np.nanmean(sgrad, axis=1), np.nan))
    npole = np.nanmean(np.where((lat > 50) & (lat < 70),
                                np.nanmean(sgrad, axis=1), np.nan))
    strat = np.nanmean(rho[-1][np.asarray(omask)]) - np.nanmean(rho[0][np.asarray(omask)])
    print(f"surf rho: eq {eq:.3f}  subpolar(50-70N) {npole:.3f}  "
          f"contrast {npole-eq:+.3f} kg/m^3   top->bot strat {strat:+.3f}")
    # N. Atlantic column: is there dense water at the surface to sink?
    na = (lat > 45) & (lat < 70)
    na_surf = np.nanmean(np.where(na[:, None] & basin, surf, np.nan))
    na_deep = np.nanmean(np.where(na[:, None] & basin, rho[-1], np.nan))
    print(f"  N.Atl(45-70N) surf rho {na_surf:.3f}  deep rho {na_deep:.3f}  "
          f"(unstable if surf>deep => convection)")

    # --- temperature/salinity ranges (clip check) ---
    T = np.asarray(st.temp); S = np.asarray(st.salt)
    print(f"  T [{T.min()-273.15:.1f},{T.max()-273.15:.1f}]C  "
          f"S [{S.min():.2f},{S.max():.2f}] psu")


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print("usage: diagnose_prognostic_amoc.py <ckpt.npz> [...]"); sys.exit(1)
    for p in paths:
        report(p)
