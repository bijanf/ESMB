"""Differentiable dinosaur<->ocean coupling primitives (P0 of the working-ESM roadmap).

The standalone control harness (``experiments/run_dino_coupled.py``) couples the
multi-level ``dinosaur`` atmosphere to the ocean with **numpy** helpers
(``ocean_fluxes`` and ``DinoAtmosphere.diagnostics`` both call ``np.asarray`` /
pint ``.magnitude``) and a Python ``for`` loop. Those numpy hops SEVER the JAX graph,
so the active (dino) model is NOT end-to-end differentiable today — the existing
``jax.grad`` demos run the *legacy* single-level path.

This module is the jnp-native re-implementation of those graph-cut points, so a
single coupling interval can be expressed as one differentiable JAX function:

    SST(lin) --regrid--> SST(Gauss) --DinoAtmosphere--> surface fields(Gauss)
    --regrid--> bulk fluxes(lin) --step_ocean(scan)--> ocean(t+1)

Everything here is pure ``jax.numpy`` (no numpy, no pint), reusing the float scale
factors precomputed in ``DinoAtmosphere.__init__`` (``v_scale_ms``, ``p_scale_Pa``;
the temperature scale is 1 K so nondimensional T == Kelvin numerically). The numpy
``DinoAtmosphere.diagnostics`` is kept as-is for *scoring* (dashboard / NetCDF); this
module is the *differentiable* path.

See memory ``working-esm-roadmap-2026-06-19`` / workflow wxf3z5fpu for the roadmap.
"""

import functools

import jax
import jax.numpy as jnp

from dinosaur import primitive_equations as pe

from chronos_esm.atmos import physics as aphys
from chronos_esm.atmos.dino_atmos import _qsat, G
from chronos_esm.config import (ALBEDO_OCEAN, OCEAN_DZ, OCEAN_GRID, RHO_WATER,
                                CP_WATER)
from chronos_esm.ocean.diagnostics import compute_amoc

# Surface bulk-flux constants (match experiments/run_dino_coupled.py).
RHO_AIR = 1.2
CD = 1.3e-3
RESTORE_TAU_DAYS = 30.0

# Linear (ocean/atmos) grid latitudes, in degrees.
LAT_LIN = jnp.linspace(-90.0, 90.0, OCEAN_GRID.nlat)


# --------------------------------------------------------------------------- #
# Differentiable latitude regridders (linear <-> dinosaur Gaussian grid).
# --------------------------------------------------------------------------- #
def make_regridders_jax(lat_gauss, lat_lin=LAT_LIN):
    """Differentiable 1-D latitude interpolators between the linear and Gaussian
    grids (longitudes coincide). Mirrors ``run_dino_coupled.make_regridders`` but
    with ``jnp.interp`` (differentiable) instead of looped ``np.interp``.

    Both ``lat_gauss`` and ``lat_lin`` must be ascending (as required by interp).
    """
    lat_gauss = jnp.asarray(lat_gauss)
    lat_lin = jnp.asarray(lat_lin)

    def lin_to_gauss(f_lin):                 # (nlat_lin, nlon) -> (nlon, nlat_gauss)
        return jax.vmap(lambda col: jnp.interp(lat_gauss, lat_lin, col),
                        in_axes=1, out_axes=0)(f_lin)

    def gauss_to_lin(f_g):                    # (nlon, nlat_gauss) -> (nlat_lin, nlon)
        return jax.vmap(lambda row: jnp.interp(lat_lin, lat_gauss, row),
                        in_axes=0, out_axes=1)(f_g)

    return lin_to_gauss, gauss_to_lin


# --------------------------------------------------------------------------- #
# Differentiable dinosaur diagnostics (jnp port of DinoAtmosphere.diagnostics).
# --------------------------------------------------------------------------- #
def dino_diagnostics_jax(atm, state):
    """Dimensional surface/3-D fields on the dinosaur nodal grid (nlon, nlat),
    as **jax** arrays (differentiable). Same physics as
    ``DinoAtmosphere.diagnostics`` but with no numpy/pint, so it composes into
    ``jax.grad``/``jax.jit``.

    Keys: u, v [m/s] (layers,..); temperature [K]; specific_humidity [kg/kg];
    u_sfc/v_sfc/t_sfc/q_sfc (bottom level); surface_pressure [Pa]; mslp [Pa];
    precip [kg/m^2/s].
    """
    coords = atm.coords
    cos_lat = jnp.asarray(atm.cos_lat)                     # (nlon, nlat)

    d = pe.compute_diagnostic_state(state=state, coords=coords)
    clu = d.cos_lat_u                                      # (2, layers, nlon, nlat)
    u = clu[0] / cos_lat[None] * atm.v_scale_ms
    v = clu[1] / cos_lat[None] * atm.v_scale_ms
    # temperature scale is 1 K (ref_temps already in K), so nondim T == Kelvin.
    T = jnp.asarray(atm.ref_temps)[:, None, None] + d.temperature_variation

    nodal_sp = jnp.exp(coords.horizontal.to_nodal(state.log_surface_pressure))
    ps_Pa = nodal_sp[0] * atm.p_scale_Pa                   # (nlon, nlat)

    q = jnp.maximum(coords.horizontal.to_nodal(state.tracers["specific_humidity"]), 0.0)

    sigma = jnp.asarray(atm.sigma)[:, None, None]
    p_lvl = sigma * ps_Pa
    qsat = _qsat(T, p_lvl)
    cond_si = jnp.maximum(q - qsat, 0.0) / (atm.tau_cond * atm.t0_seconds)   # 1/s
    precip = jnp.sum(cond_si * (ps_Pa * atm.dsigma) / G, axis=0)             # kg/m^2/s

    mslp = ps_Pa * jnp.exp(G * jnp.asarray(atm.orography_m) / (287.0 * 288.0))

    return dict(u=u, v=v, temperature=T, specific_humidity=q,
                u_sfc=u[-1], v_sfc=v[-1], t_sfc=T[-1], q_sfc=q[-1],
                surface_pressure=ps_Pa, mslp=mslp, precip=precip)


def run_atmos_interval(atm, state, sst_K_gauss, n_days=1):
    """Differentiable atmosphere advance over ``n_days`` (nondim T == K, so the
    SST nondimensionalization is identity — this bypasses the pint call in
    ``DinoAtmosphere.step`` that would otherwise complicate tracing). Returns the
    new modal state. ``atm._run`` is the jitted ImEx integrator."""
    n_steps = int(atm.steps_per_day * n_days)
    return atm._run(state, jnp.asarray(sst_K_gauss), n_steps)


# --------------------------------------------------------------------------- #
# Differentiable bulk surface fluxes (jnp port of run_dino_coupled.ocean_fluxes).
# --------------------------------------------------------------------------- #
def co2_forcing_wm2(co2_ppm, co2_ref=280.0):
    """Myhre (1998) CO2 radiative forcing F = 5.35*ln(C/C0) [W/m^2] (0 at C0).
    This is the SINGLE forcing channel: it is added to the ocean surface heat
    budget as extra downwelling longwave, NOT as an offset to the SST-slaved
    atmospheric Teq (which would be nearly inert and would double-count). Under the
    WOA flux-correction (~50 W/m2/K) the response is suppressed; freeing the ocean
    (q-flux, the next P2 step) lets F drive a real warming. Differentiable in C, so
    d(climate)/d(CO2) is available for sensitivity/calibration."""
    return 5.35 * jnp.log(jnp.asarray(co2_ppm) / co2_ref)


def ocean_fluxes_jax(sst_K, u_sfc, v_sfc, t_air_K, q_air, precip_atm, *,
                     balance_heat=True, ocean_mask=None, sst_target=None,
                     restore_tau_days=RESTORE_TAU_DAYS, lat_lin=LAT_LIN, co2_ppm=None):
    """Bulk surface fluxes on the linear grid. Returns
    ``(net_heat W/m2, fw kg/m2/s, tau_x Pa, tau_y Pa)``. With ``co2_ppm=None`` this is
    numerically identical to ``run_dino_coupled.ocean_fluxes`` (pure-jnp: the
    boolean-indexed ocean-mean removals are masked weighted sums). With ``co2_ppm``
    set, the Myhre CO2 forcing is added to the surface heat budget (single channel).
    """
    wlat = jnp.cos(jnp.deg2rad(lat_lin))[:, None]

    insol = aphys.compute_solar_insolation(lat_lin * jnp.pi / 180.0,
                                           day_of_year=80.0)[:, None]
    sw_net = jnp.maximum(insol, 0.0) * (1.0 - ALBEDO_OCEAN)
    lw_down = 0.8 * 5.67e-8 * jnp.maximum(t_air_K - 10.0, 150.0) ** 4
    lw_up = 0.98 * 5.67e-8 * sst_K ** 4
    sens, lat = aphys.compute_surface_fluxes(t_air_K, q_air, u_sfc, v_sfc, sst_K)
    net_heat = sw_net + lw_down - lw_up - sens - lat

    if co2_ppm is not None:
        # single-channel CO2 forcing into the surface heat budget (before the
        # flux-correction, which is the artifact that suppresses it).
        net_heat = net_heat + co2_forcing_wm2(co2_ppm)

    if sst_target is not None:
        lam = RHO_WATER * CP_WATER * float(OCEAN_DZ[0]) / (restore_tau_days * 86400.0)
        net_heat = net_heat + lam * (jnp.asarray(sst_target) - sst_K)
    elif balance_heat:
        w = jnp.broadcast_to(wlat, net_heat.shape)
        if ocean_mask is not None:
            m = jnp.asarray(ocean_mask).astype(bool)
            ocean_mean = jnp.sum(jnp.where(m, net_heat * w, 0.0)) / jnp.sum(jnp.where(m, w, 0.0))
        else:
            ocean_mean = jnp.sum(net_heat * w) / jnp.sum(w)
        net_heat = net_heat - ocean_mean
    net_heat = jnp.clip(net_heat, -1500.0, 1500.0)

    evap = lat / 2.5e6
    fw = precip_atm - evap
    if balance_heat and ocean_mask is not None:
        wf = jnp.broadcast_to(wlat, fw.shape)
        mf = jnp.asarray(ocean_mask).astype(bool)
        fw = fw - jnp.sum(jnp.where(mf, fw * wf, 0.0)) / jnp.sum(jnp.where(mf, wf, 0.0))

    wind_mag = jnp.maximum(jnp.sqrt(u_sfc ** 2 + v_sfc ** 2), 1.0)
    tau_x = jnp.clip(RHO_AIR * CD * wind_mag * u_sfc, -0.3, 0.3)
    tau_y = jnp.clip(RHO_AIR * CD * wind_mag * v_sfc, -0.3, 0.3)
    return net_heat, fw, tau_x, tau_y


# --------------------------------------------------------------------------- #
# Unified differentiable AMOC scalar (the metric P3 must make density-responsive).
# --------------------------------------------------------------------------- #
def amoc_strength(ocean_state, ocean_mask=None, atlantic_mask=None):
    """Differentiable scalar AMOC metric: the upper-cell strength at 26.5N [Sv],
    using the proper basin-restricted, barotropic-removed ``compute_amoc`` (NOT
    the cruder ``compute_moc`` that ``verify_gradient.py`` currently uses).

    On the present diagnostic-velocity ocean ``d(amoc_strength)/d(density) == 0``
    (verified): density has no pathway to net overturning. Breaking that to a
    non-zero, sign-correct gradient is the P3 (prognostic-momentum) exit criterion.
    """
    res = compute_amoc(ocean_state, atlantic_mask=atlantic_mask, ocean_mask=ocean_mask)
    return res["upper_cell_26N"]


__all__ = [
    "make_regridders_jax", "dino_diagnostics_jax", "run_atmos_interval",
    "ocean_fluxes_jax", "co2_forcing_wm2", "amoc_strength", "LAT_LIN",
]
