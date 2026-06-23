"""Assembled PROGNOSTIC ocean dynamics (P3 / S5c): the real ocean the owner asked
for -- prognostic momentum -> prognostic streamfunction -> density-driven overturning.

This combines the two validated building blocks into one mass-conserving step:

  * the BAROCLINIC mode (chronos_esm/ocean/momentum.py: spherical, implicitly-viscous,
    no-slip momentum) carries the density-driven OVERTURNING (the AMOC). Its depth-mean
    is removed each step, so it is purely baroclinic (the barotropic pressure gradient,
    balanced by the unknown rigid-lid surface pressure, drops out with the depth-mean).
  * the BAROTROPIC mode (chronos_esm/ocean/barotropic.py: Munk transport streamfunction)
    carries the wind-driven GYRES. Because the flow is reconstructed from a streamfunction
    psi, the depth-integrated transport is non-divergent BY CONSTRUCTION -- rigid-lid mass
    conservation is exact, no projection needed.

Total velocity  u = u_baroclinic (depth-mean 0) + u_barotropic (= transport/H, depth-uniform).

STATUS (S5c, in progress): the BAROCLINIC overturning is stable and PROGNOSTIC -- on
WOA density it gives a finite, bounded ~160 Sv AMOC (|v|~0.4 m/s) that emerges from
density, not a box formula. It is still ~8x too strong and the cell does not fully
close; reaching the realistic ~15-20 Sv needs GM eddies + A_h tuning + more vertical
layers (next). The BAROTROPIC gyre is stable with the flat-bottom operator (default);
the topographic coef=1/H is ill-conditioned on the real bathymetry and needs work, and
its explicit beta-advection needs sub-stepping (n_bt_sub) / weak bottom drag (r_bt) for
the global-grid CFL. STANDALONE (not wired into veros_driver/main); S5d wires it in.
Differentiable throughout (jnp + lax.scan).
"""

import jax
import jax.numpy as jnp

from chronos_esm.ocean import barotropic as bt
from chronos_esm.ocean.momentum import OMEGA, RHO0, step_momentum_sphere


def _depth_mean(field, dz3, mask3d, H):
    """Wet-column depth average of a 3-D field -> (ny, nx)."""
    return jnp.sum(field * dz3 * mask3d, axis=0) / H


def column_depth(dz, mask3d):
    """Wet-column depth H (ny, nx), floored at the top layer thickness to avoid /0."""
    dz3 = jnp.asarray(dz)[:, None, None]
    return jnp.maximum(jnp.sum(dz3 * mask3d, axis=0), jnp.asarray(dz).reshape(-1)[0])


def step_prognostic_dynamics(
    u,
    v,
    psi,
    zeta,
    rho,
    taux,
    tauy,
    *,
    lat,
    dlon,
    dlat,
    a,
    dz,
    dt,
    A_h_bc,
    A_h_bt,
    mask3d,
    coef=None,
    omega=OMEGA,
    rho0=RHO0,
    r_bc=0.0,
    r_bt=0.0,
    n_bt_sub=1,
    visc_iter=120,
    bt_max_iter=400,
    bt_visc_iter=120,
):
    """One step of the assembled prognostic ocean dynamics. Returns (u, v, psi, zeta).

    A_h_bc : baroclinic lateral eddy viscosity [m^2/s] (sets the interior smoothing /
             the baroclinic boundary layer of the overturning).
    A_h_bt : barotropic Munk lateral viscosity [m^2/s] (sets the gyre western boundary
             layer width (A_h_bt/beta)^(1/3))."""
    dz3 = jnp.asarray(dz)[:, None, None]
    H = column_depth(dz, mask3d)  # (ny, nx)
    surf = mask3d[0]
    if coef is None:
        # FLAT-BOTTOM operator zeta = lap(psi) (Stommel/Munk), the validated stable
        # choice. The topographic coef = 1/H (JEBAR-like) is ill-conditioned on the
        # real coarse bathymetry -- 1/H spans orders of magnitude at the coasts -- and
        # makes the global barotropic solve UNSTABLE (verified: coef=1/H -> NaN, coef=1
        # -> finite). Pass coef=1/H explicitly only once that conditioning is addressed.
        coef = jnp.ones_like(surf)

    # --- 1. baroclinic momentum (density-driven overturning) -------------------
    ubt = _depth_mean(u, dz3, mask3d, H)
    vbt = _depth_mean(v, dz3, mask3d, H)
    u_bc = (u - ubt[None]) * mask3d
    v_bc = (v - vbt[None]) * mask3d
    u_bc, v_bc = step_momentum_sphere(
        u_bc,
        v_bc,
        rho,
        lat=lat,
        dlon=dlon,
        dlat=dlat,
        a=a,
        dz=dz,
        dt=dt,
        A_h=A_h_bc,
        mask=mask3d,
        omega=omega,
        r=r_bc,
        rho0=rho0,
        visc_iter=visc_iter,
    )
    # re-remove the depth mean so the prognostic mode stays purely baroclinic
    u_bc = (u_bc - _depth_mean(u_bc, dz3, mask3d, H)[None]) * mask3d
    v_bc = (v_bc - _depth_mean(v_bc, dz3, mask3d, H)[None]) * mask3d

    # --- 2. barotropic Munk streamfunction (wind-driven gyres) -----------------
    # The barotropic mode is the fast one (Rossby/topographic waves); its explicit
    # beta-advection sets a tight CFL on the global grid, so optionally SUB-STEP it
    # n_bt_sub times per baroclinic step, with an optional weak bottom drag r_bt.
    F = (bt.wind_stress_curl_sphere(taux, tauy, lat, dlon, dlat, a) / (rho0 * H)) * surf
    dt_bt = dt / n_bt_sub

    def _bt_body(carry, _):
        zeta, psi = carry
        zeta, psi = bt.step_barotropic_munk_sphere(
            zeta,
            psi,
            F,
            lat=lat,
            dlon=dlon,
            dlat=dlat,
            a=a,
            omega=omega,
            dt=dt_bt,
            r=r_bt,
            A_h=A_h_bt,
            mask=surf,
            coef=coef,
            max_iter=bt_max_iter,
            visc_iter=bt_visc_iter,
        )
        return (zeta, psi), None

    (zeta, psi), _ = jax.lax.scan(_bt_body, (zeta, psi), None, length=n_bt_sub)
    U_bt, V_bt = bt.velocities_sphere(psi, lat, dlon, dlat, a, surf)  # transport m^2/s
    u_bt2 = (U_bt / H)[None]  # barotropic velocity (depth-uniform)
    v_bt2 = (V_bt / H)[None]

    # --- 3. total velocity = baroclinic + barotropic ---------------------------
    u_new = (u_bc + u_bt2) * mask3d
    v_new = (v_bc + v_bt2) * mask3d
    return u_new, v_new, psi, zeta


def spin_up_prognostic_dynamics(
    rho,
    taux,
    tauy,
    *,
    lat,
    dlon,
    dlat,
    a,
    dz,
    dt,
    A_h_bc,
    A_h_bt,
    mask3d,
    n_steps,
    coef=None,
    omega=OMEGA,
    rho0=RHO0,
    r_bc=0.0,
    r_bt=0.0,
    n_bt_sub=1,
    visc_iter=120,
    bt_max_iter=400,
    bt_visc_iter=120,
):
    """Spin the assembled prognostic dynamics from rest under a fixed density field +
    wind stress to (near) steady state. Returns (u, v, psi, zeta). Differentiable."""
    nz, ny, nx = rho.shape
    u = jnp.zeros_like(rho)
    v = jnp.zeros_like(rho)
    psi = jnp.zeros((ny, nx), rho.dtype)
    zeta = jnp.zeros((ny, nx), rho.dtype)

    def body(carry, _):
        u, v, psi, zeta = carry
        u, v, psi, zeta = step_prognostic_dynamics(
            u,
            v,
            psi,
            zeta,
            rho,
            taux,
            tauy,
            lat=lat,
            dlon=dlon,
            dlat=dlat,
            a=a,
            dz=dz,
            dt=dt,
            A_h_bc=A_h_bc,
            A_h_bt=A_h_bt,
            mask3d=mask3d,
            coef=coef,
            omega=omega,
            rho0=rho0,
            r_bc=r_bc,
            r_bt=r_bt,
            n_bt_sub=n_bt_sub,
            visc_iter=visc_iter,
            bt_max_iter=bt_max_iter,
            bt_visc_iter=bt_visc_iter,
        )
        return (u, v, psi, zeta), None

    (u, v, psi, zeta), _ = jax.lax.scan(body, (u, v, psi, zeta), None, length=n_steps)
    return u, v, psi, zeta


__all__ = [
    "step_prognostic_dynamics",
    "spin_up_prognostic_dynamics",
    "column_depth",
]
