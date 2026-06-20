"""
JAX-based driver for the Ocean component (Veros-like physics).
Reliable v8 version with Stommel Solver and RK4 Tracers.
"""

from functools import partial
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from chronos_esm.ocean.utils import soft_clip

from chronos_esm.config import (  # noqa: F401
    DT_OCEAN,
    GRAVITY,
    OCEAN_GRID,
    OMEGA,
    RHO_WATER,
)
from chronos_esm.ocean import mixing, solver, overturning


class OceanState(NamedTuple):
    """State of the ocean model."""

    u: jnp.ndarray  # Zonal velocity (nz, ny, nx)
    v: jnp.ndarray  # Meridional velocity (nz, ny, nx)
    w: jnp.ndarray  # Vertical velocity (nz, ny, nx)
    temp: jnp.ndarray  # Potential temperature (nz, ny, nx)
    salt: jnp.ndarray  # Salinity (nz, ny, nx)
    psi: jnp.ndarray  # Barotropic streamfunction (ny, nx)
    rho: jnp.ndarray  # Density (nz, ny, nx)
    dic: jnp.ndarray  # Dissolved Inorganic Carbon [mmol/m^3] (nz, ny, nx)


def equation_of_state(temp: jnp.ndarray, salt: jnp.ndarray) -> jnp.ndarray:
    """
    Linear equation of state for seawater.
    """
    rho_0 = RHO_WATER
    alpha = 0.2
    beta = 0.8
    t0 = 283.15
    s0 = 35.0
    return rho_0 - alpha * (temp - t0) + beta * (salt - s0)


@partial(jax.jit, static_argnames=["nz", "ny", "nx", "dt"])
def step_ocean(
    state: OceanState,
    surface_fluxes: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    wind_stress: Tuple[jnp.ndarray, jnp.ndarray],
    dx: jnp.ndarray,
    dy: float,
    dz: jnp.ndarray,
    nz: int = OCEAN_GRID.nz,
    ny: int = OCEAN_GRID.nlat,
    nx: int = OCEAN_GRID.nlon,
    mask: Optional[jnp.ndarray] = None,
    dt: float = DT_OCEAN,
    r_drag: float = 5.0e-2,
    kappa_gm: float = 1000.0,
    kappa_h: float = 500.0,
    kappa_bi: float = 0.0,
    Ah: float = 2.0e5,
    Ab: float = 0.0,
    shapiro_strength: float = 0.0,
    smag_constant: float = 0.1,
    ocean_mask_3d: Optional[jnp.ndarray] = None,
    thc_k_vel: float = 1.0e-4,
    thc_haline_gain: float = 1.0,
    hosing_sv: float = 0.0,
) -> OceanState:

    # Helpers
    def compute_shapiro_filter(field, strength):
        # Scale-selective (biharmonic, del^4) filter: strongly damps 2-grid-point
        # checkerboard noise while leaving resolved gradients nearly untouched.
        # The previous del^2 form (field + strength/4 * laplacian) applied EVERY
        # step was a strong low-pass smoother that erased real SST/SSS gradients
        # (WOA structure collapsed to ~uniform within days). Primary grid-noise
        # control is the physical Laplacian diffusion (kappa_h on tracers, Ah on
        # momentum); this filter is a light 2dx cleanup, off by default.
        def _lap(f):
            d2x = jnp.roll(f, -1, axis=2) - 2 * f + jnp.roll(f, 1, axis=2)
            fp = jnp.pad(f, ((0, 0), (1, 1), (0, 0)), mode="edge")
            d2y = fp[:, 2:, :] - 2 * f + fp[:, :-2, :]
            return d2x + d2y
        # del^4 = lap(lap); /16 so strength~0.25 fully removes a 2dx mode.
        return field - (strength / 16.0) * _lap(_lap(field))

    # 1. Update Density
    rho = equation_of_state(state.temp, state.salt)

    # 2. Mixing & Bolus
    ny_idx = jnp.arange(ny)
    is_pole = (ny_idx < 5) | (ny_idx >= ny - 5)
    interior_mask_3d = jnp.where(is_pole, 0.0, 1.0)[None, :, None]
    pole_mask_3d = 1.0 - interior_mask_3d
    
    sx, sy = mixing.compute_isopycnal_slopes(rho, dx, dy, dz)
    kappa_gm_eff = kappa_gm * interior_mask_3d
    u_bolus, v_bolus, _ = mixing.compute_gm_bolus_velocity(kappa_gm_eff, sx, sy, dz)
    u_eff, v_eff = state.u + u_bolus, state.v + v_bolus

    # 3. Barotropic Streamfunction (Stommel)
    tau_x, tau_y = wind_stress
    curl_tau = (jnp.roll(tau_y, -1, axis=1) - jnp.roll(tau_y, 1, axis=1)) / (2 * dx) - (
        jnp.roll(tau_x, -1, axis=0) - jnp.roll(tau_x, 1, axis=0)
    ) / (2 * dy)
    
    H_total = jnp.sum(dz)
    r_drag_bt = 1.0 / (86400.0 * 25.0) # Tuned 25-day drag
    rhs_psi = curl_tau / (RHO_WATER * H_total * r_drag_bt)

    psi_new, _ = solver.solve_poisson_2d(rhs_psi, dx, dy, max_iter=100, tol=1e-4, mask=mask, x0=state.psi)
    u_bt = -(jnp.roll(psi_new, -1, axis=0) - jnp.roll(psi_new, 1, axis=0)) / (2 * dy)
    v_bt = (jnp.roll(psi_new, -1, axis=1) - jnp.roll(psi_new, 1, axis=1)) / (2 * dx)

    # 4. Baroclinic (Thermal Wind)
    rho_anom = rho - RHO_WATER
    dz_3d = dz.reshape(-1, 1, 1)
    pressure_hydro = jnp.cumsum(GRAVITY * rho_anom * dz_3d, axis=0)
    dp_dx = (jnp.roll(pressure_hydro, -1, axis=2) - jnp.roll(pressure_hydro, 1, axis=2)) / (2 * dx)
    dp_dy = (jnp.roll(pressure_hydro, -1, axis=1) - jnp.roll(pressure_hydro, 1, axis=1)) / (2 * dy)
    
    lat_rad = jnp.deg2rad(jnp.linspace(-90, 90, ny))
    f_cor = 2 * OMEGA * jnp.sin(lat_rad)
    f_3d = jnp.broadcast_to(f_cor[None, :, None], (nz, ny, nx))
    f_clamped = jnp.where(jnp.abs(f_3d) < 1e-5, jnp.sign(f_3d + 1e-16) * 1e-5, f_3d)
    
    denom = RHO_WATER * (f_clamped**2 + r_drag**2)
    u_geo = -(r_drag * dp_dx + f_clamped * dp_dy) / denom
    v_geo = (f_clamped * dp_dx - r_drag * dp_dy) / denom
    
    # --- Bathymetry wet masks: enforce no-flux at coasts and the sea floor ---
    # maskC = 1 in wet cells, 0 in land/below-floor. Face masks are open only if
    # BOTH adjacent centers are wet (MITgcm hFac MIN rule), so no advective or
    # diffusive flux is ever exchanged with a dry cell. All static -> AD-safe.
    # With no mask supplied the whole domain is wet (legacy flat-bottom behaviour).
    if ocean_mask_3d is not None:
        maskC = ocean_mask_3d.astype(u_geo.dtype)
    else:
        maskC = jnp.ones((nz, ny, nx), u_geo.dtype)
    surf2d = maskC[0]
    fW = maskC * jnp.roll(maskC, 1, 2)
    fE = maskC * jnp.roll(maskC, -1, 2)
    fS = (maskC * jnp.roll(maskC, 1, 1)).at[:, 0, :].set(0.0)
    fN = (maskC * jnp.roll(maskC, -1, 1)).at[:, -1, :].set(0.0)
    # vertical interface below cell k is open only if k+1 is also wet (floor=no-flux)
    iface = (maskC * jnp.concatenate([maskC[1:], jnp.zeros_like(maskC[:1])], 0))[:-1]
    wet_dz = dz_3d * maskC
    H_col = jnp.maximum(jnp.sum(wet_dz, axis=0), dz[0])  # per-column wet depth [m]

    # Baroclinic anomaly: remove the WET-column vertical mean (per-column depth).
    u_bar = jnp.sum(u_geo * wet_dz, axis=0) / H_col
    v_bar = jnp.sum(v_geo * wet_dz, axis=0) / H_col
    u_bc = (u_geo - u_bar[None, :, :]) * maskC
    v_bc = (v_geo - v_bar[None, :, :]) * maskC

    u_new = (u_bt[None, :, :] + u_bc) * maskC
    v_new = (v_bt[None, :, :] + v_bc) * maskC

    # --- Barotropic mass-conservation corrector ------------------------------------
    # A flat-bottom velocity streamfunction (Stommel) applied over VARIABLE bathymetry
    # leaves a large spurious NET meridional transport: the depth-integrated transport
    # is v_bt*H_col, and d(psi*H)/dx != H*d(psi)/dx, so sum_x,z v*dx*dz is ~+-350 Sv
    # where a closed ocean requires ~0 -> a clean AMOC overturning is impossible. As a
    # first correction (roadmap step 1) remove the per-latitude net: subtract a
    # latitude-uniform meridional velocity so the zonally+vertically integrated
    # transport vanishes at every latitude. dx is uniform in x, so it cancels in the
    # ratio. (A per-basin / C-grid transport streamfunction is the fuller fix.)
    col_dz = dz_3d * maskC
    net_v = jnp.sum(v_new * col_dz, axis=(0, 2))          # (ny,)  ~ net transport / dx
    area_v = jnp.sum(col_dz, axis=(0, 2)) + 1e-20         # (ny,)  wet x-z area / dx
    v_new = (v_new - (net_v / area_v)[None, :, None]) * maskC

    # --- Thermohaline overturning closure (P3/S1): density-driven Atlantic cell ----
    # A depth-integral-zero meridional overturning whose strength scales with the
    # subpolar-minus-subtropical upper-ocean density contrast (denser subpolar ->
    # stronger AMOC). Depth-integral-zero => it carries no net transport (survives the
    # corrector above) and only drives the overturning + its w/tracer transport. This
    # gives DENSITY a pathway to the 26.5N cell, so d(AMOC)/d(density) is nonzero and
    # sign-correct (the diagnostic thermal wind alone gave ~0). Interim box-model-style
    # closure; the prognostic-momentum/JEBAR core is P3 S2-S5. Disable with thc_k_vel=0.
    v_thc, _, _ = overturning.thc_overturning_velocity(
        rho, state.salt, dz, maskC, k_vel=thc_k_vel, haline_gain=thc_haline_gain)
    v_new = (v_new + v_thc) * maskC

    u_eff = u_eff * maskC
    v_eff = (v_eff + v_thc) * maskC

    # --- Vertical velocity for tracer advection (continuity, rigid lid) -----------
    # An overturning circulation only transports heat/salt if w ADVECTS tracers; a
    # diffusion-only column cannot sustain a deep cell (dense high-lat water never
    # fills the abyss and the interior diffusive-upwelling return branch is dead ->
    # no coherent AMOC). Diagnose w from the divergence of the SAME horizontal flow
    # that advects tracers (u_eff, v_eff, incl. the GM bolus) so the 3-D velocity is
    # nondivergent and vertical advection conserves the tracer integral. w_if[k] =
    # vertical velocity (z up) at the TOP interface of layer k: integrate continuity
    # (dw/dz = -div_h) UP from the rigid floor (w=0); the surface interface is closed
    # (rigid lid: no tracer advected through the ocean surface).
    dudx_e = (jnp.roll(u_eff, -1, axis=2) - jnp.roll(u_eff, 1, axis=2)) / (2 * dx)
    v_eff_pad = jnp.pad(v_eff, ((0, 0), (1, 1), (0, 0)), mode="edge")
    dvdy_e = (v_eff_pad[:, 2:, :] - v_eff_pad[:, :-2, :]) / (2 * dy)
    div_e = (dudx_e + dvdy_e) * maskC
    # Rigid-lid projection: remove the depth-MEAN horizontal divergence so the
    # depth-INTEGRATED flow is nondivergent (=> w = 0 at BOTH the surface and the
    # floor). This makes the discrete 3-D advecting flow (u_eff, v_eff, w) exactly
    # nondivergent, so the advective-form vertical advection CONSERVES the tracer
    # mean. Without it the leftover column-divergence residual sits entirely in the
    # surface cell and leaks the global mean (~+0.25 K/yr spurious heating) since the
    # advective form (unlike flux form) has no compensating compression term there.
    col_div = jnp.sum(div_e * dz_3d, axis=0)                          # (ny,nx) integrated
    div_e = (div_e - (col_div / H_col)[None, :, :]) * maskC
    w_col = -jnp.cumsum((div_e * dz_3d)[::-1], axis=0)[::-1]          # (nz,) top-of-layer
    # interior interface open only where both adjacent centres are wet; surface/floor closed
    iface_open = jnp.concatenate(
        [jnp.zeros((1, ny, nx)), maskC[:-1] * maskC[1:], jnp.zeros((1, ny, nx))], axis=0)
    w_if = jnp.concatenate(
        [jnp.zeros((1, ny, nx)), w_col[1:], jnp.zeros((1, ny, nx))], axis=0) * iface_open

    # 5. Tracers (RK4) with flux-masked advection/diffusion (no-flux at coast/floor)
    heat_flux, fw_flux, dic_flux = surface_fluxes
    # Surface fluxes enter only the top WET cell.
    fT_s = heat_flux / (RHO_WATER * 3985. * dz[0]) * surf2d
    fS_s = -fw_flux * 35. / (RHO_WATER * dz[0]) * surf2d
    # AMOC hosing: extra subpolar-N-Atlantic freshwater forcing (Sv), added directly
    # so it bypasses the surface-flux ocean-mean balancing in the coupler (which would
    # otherwise cancel the net input). The volume-mean salt renorm later only shifts
    # the absolute mean, preserving the subpolar-subtropical gradient the THC reads.
    fS_s = fS_s + overturning.subpolar_hosing_salt_tendency(hosing_sv, dx, dy, dz[0], surf2d)
    fD_s = dic_flux / dz[0] * surf2d
    k_z = mixing.compute_vertical_diffusivity(rho, dz, dt=dt)
    diff_coef = 20000. * pole_mask_3d + kappa_h * interior_mask_3d

    def _nbrs(F):
        # Neighbour value across each face; a CLOSED face returns the centre value
        # (zero gradient) so neither advection nor diffusion exchanges with dry cells.
        FW = jnp.where(fW > 0, jnp.roll(F, 1, 2), F)
        FE = jnp.where(fE > 0, jnp.roll(F, -1, 2), F)
        FS = jnp.where(fS > 0, jnp.roll(F, 1, 1), F)
        FN = jnp.where(fN > 0, jnp.roll(F, -1, 1), F)
        return FW, FE, FS, FN

    def _horiz(F):
        FW, FE, FS, FN = _nbrs(F)
        dF_dx = jnp.where(u_eff > 0, F - FW, FE - F) / dx
        dF_dy = jnp.where(v_eff > 0, F - FS, FN - F) / dy
        adv = -(u_eff * dF_dx + v_eff * dF_dy)
        lap = ((FE - F) + (FW - F)) / dx**2 + ((FN - F) + (FS - F)) / dy**2
        return adv + lap * diff_coef

    def _vdiff(F):
        dist = 0.5 * (dz_3d[:-1] + dz_3d[1:])
        grad = (F[1:] - F[:-1]) / dist
        flux = -k_z[1:-1] * grad * iface  # zero flux through dry interfaces / the floor
        flux = jnp.concatenate([jnp.zeros((1, ny, nx)), flux, jnp.zeros((1, ny, nx))], axis=0)
        return (flux[:-1] - flux[1:]) / dz_3d

    def _vadv(F):
        # Upwind vertical advection in ADVECTIVE form (-w dF/dz, z up), CONSISTENT with
        # the horizontal advective form so there is NO spurious -F*dw/dz compression
        # source (flux form mixed with advective horizontal blows T up exponentially).
        # CFL = |w|*dt/dz ~ 1e-3 (w ~ 1e-5 m/s), so explicit upwind is comfortably stable.
        dist = 0.5 * (dz_3d[:-1] + dz_3d[1:])         # (nz-1,) centre-centre distance
        open_int = maskC[:-1] * maskC[1:]             # interior interface wet on both sides
        grad_int = (F[:-1] - F[1:]) / dist * open_int  # z-up gradient across interface k|k+1
        z = jnp.zeros((1, ny, nx))
        grad_below = jnp.concatenate([grad_int, z], axis=0)   # cell k looks DOWN to k+1
        grad_above = jnp.concatenate([z, grad_int], axis=0)   # cell k looks UP to k-1
        w_c = 0.5 * (w_if[:-1] + w_if[1:])            # (nz,) cell-centred vertical velocity
        grad_up = jnp.where(w_c > 0, grad_below, grad_above)  # upwind on w sign
        return -w_c * grad_up

    def tendencies(T, S, D):
        res_T = (_horiz(T) + _vadv(T) + _vdiff(T)).at[0].add(fT_s)
        res_S = (_horiz(S) + _vadv(S) + _vdiff(S)).at[0].add(fS_s)
        res_D = (_horiz(D) + _vadv(D) + _vdiff(D)).at[0].add(fD_s)
        return res_T * maskC, res_S * maskC, res_D * maskC  # freeze dry cells

    k1_T, k1_S, k1_D = tendencies(state.temp, state.salt, state.dic)
    k2_T, k2_S, k2_D = tendencies(state.temp+0.5*dt*k1_T, state.salt+0.5*dt*k1_S, state.dic+0.5*dt*k1_D)
    k3_T, k3_S, k3_D = tendencies(state.temp+0.5*dt*k2_T, state.salt+0.5*dt*k2_S, state.dic+0.5*dt*k2_D)
    k4_T, k4_S, k4_D = tendencies(state.temp+dt*k3_T, state.salt+dt*k3_S, state.dic+dt*k3_D)
    
    temp_new = compute_shapiro_filter(state.temp + (dt/6.)*(k1_T+2*k2_T+2*k3_T+k4_T), shapiro_strength)
    salt_new = compute_shapiro_filter(state.salt + (dt/6.)*(k1_S+2*k2_S+2*k3_S+k4_S), shapiro_strength)
    dic_new = state.dic + (dt/6.)*(k1_D+2*k2_D+2*k3_D+k4_D)

    # SALINITY RELAXATION - Two-tier approach:
    # 1. Strong high-latitude sponge (6-month) to stabilize polar regions
    # 2. Weak global relaxation (5-year) as a safety net against runaway drift
    # The 5-year timescale is ~5x weaker than the original 1-year, allowing
    # meaningful density contrasts to develop for AMOC while preventing the
    # monotonic drift to clamp values seen with sponge-only.
    S_REF = 35.0  # Reference salinity [psu]

    # Tier 1: High-latitude sponge (strong, 6-month timescale)
    tau_sponge = 180.0 * 86400.0  # 6-month [s]
    sponge_rate = dt / tau_sponge
    lat_ocn = jnp.linspace(-90, 90, ny)
    sponge_weight = jnp.clip((jnp.abs(lat_ocn) - 65.0) / 5.0, 0.0, 1.0)
    sponge_3d = sponge_weight[None, :, None]
    salt_new = salt_new + sponge_rate * sponge_3d * (S_REF - salt_new)

    # Tier 2: Moderate global relaxation (3-year timescale)
    # 3-year is a compromise: weaker than original 1-year (preserves density
    # contrasts) but strong enough to prevent the runaway drift seen with 5-year.
    tau_global = 3.0 * 365.0 * 86400.0  # 3 years [s]
    global_rate = dt / tau_global
    salt_new = salt_new + global_rate * (S_REF - salt_new)

    # Hard clip for physics stability. Gradients flow via straight-through
    # estimator: jnp.clip has subgradient 1 in interior, 0 at boundaries.
    # This is sufficient for optimization since values rarely hit boundaries
    # when parameters are in reasonable ranges.
    salt_new = jnp.clip(salt_new, 30.0, 38.0)
    temp_new = jnp.clip(temp_new, 250.0, 320.0)
    # Seawater freezing floor (simple sea-ice stub): water cannot cool below the
    # ~ -1.8 C freezing point -- in the real ocean sea ice forms and its latent
    # heat of fusion halts further cooling, holding the water at the freezing
    # point. With no prognostic sea-ice model here, floor ocean temperature at
    # the freezing point so high-latitude cells cannot drift unphysically cold
    # (the deep ocean is well above this, so only cold polar surface cells are
    # affected). T_f ~= -0.054*S; -1.8 C (271.35 K) is the ~35 psu value.
    temp_new = jnp.maximum(temp_new, 271.35)

    # Enforce global-mean salinity conservation. Salt is exactly conserved in the
    # real ocean, but sea-ice brine rejection (as ice grows), the [30,38] clip,
    # and advection numerics cause a slow global-mean drift that otherwise runs
    # salinity into the clip over a long run. Renormalize the volume-weighted,
    # ocean-only global mean back to the observed reference each step -- a gentle
    # additive shift that removes the net drift without altering the spatial
    # (gradient) structure that drives the circulation.
    S_REF_GLOBAL = 34.7  # observed global-mean ocean salinity [psu]
    vol = dz_3d * maskC  # wet-volume weighting (per-column depth aware)
    s_mean = jnp.sum(salt_new * vol) / (jnp.sum(vol) + 1e-12)
    salt_new = salt_new - (s_mean - S_REF_GLOBAL) * maskC  # shift wet cells only

    # Diagnose vertical velocity from continuity (previously left at zeros).
    # Hydrostatic continuity: dw/dz = -(du/dx + dv/dy). With a rigid bottom
    # (w=0 at the ocean floor) integrate the horizontal divergence upward, so
    # w at the top of layer k = -sum_{j>=k} (div_h_j * dz_j). w[0] (surface) is
    # then the net column divergence, ~0 for a non-divergent column.
    # NOTE: uses the same Cartesian divergence as the rest of the driver (no
    # cos(lat) metric term yet); diagnostic only -- tracers are not vertically
    # advected by w here.
    dudx = (jnp.roll(u_new, -1, axis=2) - jnp.roll(u_new, 1, axis=2)) / (2 * dx)
    v_pad = jnp.pad(v_new, ((0, 0), (1, 1), (0, 0)), mode="edge")
    dvdy = (v_pad[:, 2:, :] - v_pad[:, :-2, :]) / (2 * dy)
    div_h = dudx + dvdy
    w_flux = div_h * dz_3d
    w_new = (-jnp.cumsum(w_flux[::-1], axis=0)[::-1]) * maskC

    return OceanState(u=u_new, v=v_new, w=w_new, temp=temp_new, salt=salt_new, psi=psi_new, rho=equation_of_state(temp_new, salt_new), dic=dic_new)

def init_ocean_state(nz, ny, nx) -> OceanState:
    return OceanState(u=jnp.zeros((nz, ny, nx)), v=jnp.zeros((nz, ny, nx)), w=jnp.zeros((nz, ny, nx)),
                      temp=jnp.ones((nz, ny, nx)) * (10.0 + 273.15), salt=jnp.ones((nz, ny, nx)) * 35.0,
                      psi=jnp.zeros((ny, nx)), rho=jnp.zeros((nz, ny, nx)), dic=jnp.ones((nz, ny, nx)) * 2000.0)