"""
Atmospheric Dynamics Module (Primitive Equations).

This module implements the Spectral Primitive Equations on a sphere (approximated).
It solves for:
- Vorticity (zeta)
- Divergence (div)
- Temperature (T)
- Log Surface Pressure (ln_ps)
- Specific Humidity (q) [Advected]
- CO2 (co2) [Advected]

Method:
- Spectral (FFT) in Longitude
- Finite Difference in Latitude
- Helmholtz Decomposition for U, V
"""

from functools import partial
from typing import List, NamedTuple, Optional, Tuple, Union  # noqa: F401

import jax
import jax.numpy as jnp
import numpy as np

from chronos_esm.atmos import physics, spectral
from chronos_esm.config import ATMOS_GRID, DT_ATMOS, EARTH_RADIUS
from chronos_esm.ocean.solver import jacobi_preconditioner, solve_cg, solve_poisson_2d

# Constants
R_EARTH = EARTH_RADIUS
OMEGA = 7.292e-5  # Rotation rate of Earth
GRAVITY = 9.81
RD = 287.0  # Gas constant for dry air
CP = 1004.0  # Specific heat at constant pressure
KAPPA = RD / CP


# ============================================================================
# ENERGY/ENSTROPHY CONSERVATION FIXERS
#
# Forward Euler time-stepping does not conserve energy or enstrophy.
# Over 100-year integrations, even tiny errors compound to instability.
# These fixers explicitly enforce conservation after each time step.
# Reference: Arakawa & Lamb (1981), Sadourny (1975)
# ============================================================================

def compute_total_energy(temp: jnp.ndarray, u: jnp.ndarray, v: jnp.ndarray,
                         cos_lat: jnp.ndarray) -> jnp.ndarray:
    """
    Compute total energy (kinetic + internal).
    E = ∫ (0.5*(u² + v²) + CP*T) * cos(lat) dA

    Args:
        temp: Temperature [K]
        u, v: Wind components [m/s]
        cos_lat: Cosine of latitude (ny, 1)

    Returns:
        Total energy (scalar)
    """
    kinetic = 0.5 * (u**2 + v**2)
    internal = CP * temp
    # Weight by cos(lat) for area-weighted integral
    energy_density = (kinetic + internal) * cos_lat
    return jnp.sum(energy_density)


def compute_enstrophy(vorticity: jnp.ndarray, cos_lat: jnp.ndarray) -> jnp.ndarray:
    """
    Compute total enstrophy.
    Z = ∫ ζ² * cos(lat) dA

    Args:
        vorticity: Relative vorticity [1/s]
        cos_lat: Cosine of latitude (ny, 1)

    Returns:
        Total enstrophy (scalar)
    """
    return jnp.sum(vorticity**2 * cos_lat)


def apply_energy_fixer(temp_new: jnp.ndarray, u_new: jnp.ndarray, v_new: jnp.ndarray,
                       temp_old: jnp.ndarray, u_old: jnp.ndarray, v_old: jnp.ndarray,
                       cos_lat: jnp.ndarray, alpha: float = 0.1) -> jnp.ndarray:
    """
    Apply energy conservation fixer to temperature field.

    We adjust temperature to conserve total energy. This is the standard
    approach in climate models (Williamson et al. 1992, NCAR tech note).

    The fixer blends with original to prevent overcorrection:
    T_fixed = T_new + alpha * delta_T

    Args:
        temp_new: New temperature after time step [K]
        u_new, v_new: New winds after time step [m/s]
        temp_old: Temperature before time step [K]
        u_old, v_old: Winds before time step [m/s]
        cos_lat: Cosine of latitude
        alpha: Blending factor (0.1 = gentle correction)

    Returns:
        Temperature with energy conservation correction
    """
    E_old = compute_total_energy(temp_old, u_old, v_old, cos_lat)
    E_new = compute_total_energy(temp_new, u_new, v_new, cos_lat)

    # Energy change
    dE = E_new - E_old

    # Total area weight for normalization
    total_weight = jnp.sum(cos_lat) * temp_new.shape[1]  # sum(cos_lat) * nlon

    # Temperature correction: dE = CP * delta_T * total_weight
    # delta_T = dE / (CP * total_weight)
    delta_T = dE / (CP * total_weight + 1e-10)

    # Apply gentle correction (alpha < 1 prevents overcorrection)
    temp_fixed = temp_new - alpha * delta_T

    return temp_fixed


def apply_enstrophy_fixer(vort_new: jnp.ndarray, vort_old: jnp.ndarray,
                          cos_lat: jnp.ndarray, alpha: float = 0.05) -> jnp.ndarray:
    """
    Apply enstrophy conservation fixer to vorticity field.

    Enstrophy (integral of vorticity squared) conservation prevents
    spurious cascades in the vorticity field that lead to grid-scale noise.

    We rescale vorticity to maintain total enstrophy:
    ζ_fixed = ζ_new * sqrt(Z_old / Z_new)

    But apply gently with blending factor alpha.

    Args:
        vort_new: New vorticity after time step [1/s]
        vort_old: Vorticity before time step [1/s]
        cos_lat: Cosine of latitude
        alpha: Blending factor (0.05 = very gentle)

    Returns:
        Vorticity with enstrophy conservation correction
    """
    Z_old = compute_enstrophy(vort_old, cos_lat)
    Z_new = compute_enstrophy(vort_new, cos_lat)

    # Prevent division by zero
    Z_new_safe = jnp.maximum(Z_new, 1e-20)

    # Scaling factor to conserve enstrophy
    scale = jnp.sqrt(Z_old / Z_new_safe)

    # Limit scaling to prevent extreme corrections
    scale = jnp.clip(scale, 0.9, 1.1)

    # Apply gentle correction
    vort_fixed = vort_new * (1.0 - alpha + alpha * scale)

    return vort_fixed


class AtmosState(NamedTuple):
    """State of the atmosphere (Primitive Equations)."""

    vorticity: jnp.ndarray  # Relative Vorticity [1/s]
    divergence: jnp.ndarray  # Divergence [1/s]
    temp: jnp.ndarray  # Temperature [K]
    ln_ps: jnp.ndarray  # Log Surface Pressure [log(Pa)]
    q: jnp.ndarray  # Specific Humidity [kg/kg]
    co2: jnp.ndarray  # CO2 Concentration [ppm]

    # Diagnostic variables (computed from prognostic)
    u: jnp.ndarray  # Zonal Wind [m/s]
    v: jnp.ndarray  # Meridional Wind [m/s]
    psi: jnp.ndarray  # Streamfunction
    chi: jnp.ndarray  # Velocity Potential
    phi_s: jnp.ndarray  # Surface Geopotential [m^2/s^2]


def init_atmos_state(
    ny: int = ATMOS_GRID.nlat, nx: int = ATMOS_GRID.nlon
) -> AtmosState:
    """Initialize the atmospheric state."""
    # Grid
    lat = jnp.linspace(-90, 90, ny)
    # lon = jnp.linspace(0, 360, nx, endpoint=False)
    lat_rad = jnp.deg2rad(lat)

    # Topography (Real ETOPO) - Load FIRST for Hydrostatic Init
    try:
        from chronos_esm import data
        # data.load_topography returns elevation in meters
        topo_m = data.load_topography(ny, nx)
        phi_s = topo_m * GRAVITY
    except Exception as e:
        print(f"Warning: Failed to load real topography ({e}), falling back to flat.")
        phi_s = jnp.zeros((ny, nx))

    # Initial Conditions (Jablonowski-Williamson inspired basic state)
    u0 = 10.0 * jnp.cos(lat_rad)[:, None] * jnp.ones((ny, nx))  # Zonal jet (Reduced to 10 m/s to prevent latent heat shock)
    v0 = jnp.zeros((ny, nx))

    # Vorticity and Divergence from U, V
    # We need gradients.
    dx = 2 * jnp.pi * R_EARTH * jnp.cos(lat_rad)[:, None] / nx
    dy = jnp.pi * R_EARTH / ny

    # Simple FD for initialization
    dv_dx, dv_dy = spectral.compute_gradients(v0, dx, dy)
    du_dx, du_dy = spectral.compute_gradients(u0, dx, dy)

    vorticity = dv_dx - du_dy
    divergence = du_dx + dv_dy

    # Add random perturbation to vorticity to trigger instability
    # Use a fixed key for reproducibility of the "default" state
    key = jax.random.PRNGKey(42)
    noise = jax.random.uniform(key, shape=(ny, nx), minval=-1e-6, maxval=1e-6)
    vorticity = vorticity + noise

    temp = 300.0 - 20.0 * jnp.sin(lat_rad)[:, None] ** 2  # Warm equator, cold poles
    temp = jnp.broadcast_to(temp, (ny, nx))

    # Initialize Surface Pressure Hydrostatically
    # Ps = P0 * exp(-Phi_s / (R * T))
    # This prevents massive pressure gradients on mountain slopes
    p0 = 101325.0
    # Use the initialized temperature for the scale height
    ln_ps = jnp.log(p0 * jnp.exp(-phi_s / (RD * temp)))

    # Initialize q with Relative Humidity = 0.8
    # This prevents massive initial evaporation from dry air
    pressure = np.exp(ln_ps) # 101325
    q_sat = physics.compute_saturation_humidity(temp, pressure)
    q = 0.8 * q_sat

    co2 = 400.0 * jnp.ones((ny, nx))
    
    return AtmosState(
        vorticity=vorticity,
        divergence=divergence,
        temp=temp,
        ln_ps=ln_ps,
        q=q,
        co2=co2,
        u=u0,
        v=v0,
        psi=jnp.zeros((ny, nx)),
        chi=jnp.zeros((ny, nx)),
        phi_s=phi_s
    )


@partial(jax.jit, static_argnames=["ny", "nx"])
def solve_poisson_sphere(
    rhs: jnp.ndarray, dx: jnp.ndarray, dy: float, ny: int, nx: int  # (ny, 1)
) -> jnp.ndarray:
    """
    Solve ∇²ψ = rhs on a sphere (channel approximation).
    BC: Periodic in X, Dirichlet=0 at Y boundaries (Poles).
    """
    # Flatten
    b = -rhs.flatten()
    x0 = jnp.zeros_like(b)

    # Operator
    def laplacian_operator(x_flat):
        x_2d = x_flat.reshape((ny, nx))

        # X: Periodic
        d2x = (
            jnp.roll(x_2d, 1, axis=1) - 2 * x_2d + jnp.roll(x_2d, -1, axis=1)
        ) / dx**2

        # Y: Dirichlet (0 at boundaries)
        # Pad with 0
        x_padded = jnp.pad(x_2d, ((1, 1), (0, 0)), mode="constant", constant_values=0)
        d2y = (x_padded[2:, :] - 2 * x_2d + x_padded[:-2, :]) / dy**2

        lap = d2x + d2y
        return -lap.flatten()

    # Preconditioner (Diagonal)
    diag_val = 2.0 / dx**2 + 2.0 / dy**2
    diag_val = jnp.broadcast_to(diag_val, (ny, nx))
    precond = jacobi_preconditioner(diag_val.flatten())

    # Solve
    x_flat, _ = solve_cg(
        laplacian_operator, b, x0, max_iter=200, tol=1e-5, preconditioner=precond
    )

    return x_flat.reshape((ny, nx))


@partial(jax.jit, static_argnames=["ny", "nx"])
def step_atmos(
    state: AtmosState,
    surface_temp: jnp.ndarray,  # SST or Land Temp
    flux_sensible: jnp.ndarray,  # W/m2
    flux_latent: jnp.ndarray,  # W/m2
    flux_co2: jnp.ndarray,  # ppm/s
    sw_down: Optional[jnp.ndarray] = None, # Downward shortwave flux (Seasonal)
    solar_constant: float = 1361.0,
    physics_params: Optional[dict] = None, # Differentiable params
    dt: float = DT_ATMOS,
    ny: int = ATMOS_GRID.nlat,
    nx: int = ATMOS_GRID.nlon,
) -> Tuple[
    AtmosState, Tuple[jnp.ndarray, jnp.ndarray]
]:  # Returns State, (Precip, SfcPressure)
    """
    Time step the atmosphere (Dynamics + Physics).
    """
    # Grid metrics
    lat = jnp.linspace(-90, 90, ny)
    lat_rad = jnp.deg2rad(lat)
    cos_lat = jnp.cos(lat_rad)[:, None]
    sin_lat = jnp.sin(lat_rad)[:, None]

    # Avoid division by zero at poles
    cos_lat = jnp.where(cos_lat < 1e-5, 1e-5, cos_lat)

    # CFL-safe dx for ALL horizontal derivatives (advection, gradients, pressure
    # gradient, Laplacian) -- not just diffusion. The bare lat-lon dx collapses to
    # ~4 m at the pole rows (cos_lat floored to 1e-5), so any zonal structure makes
    # d/dx and d^2/dx^2 explode: advect(ln_ps) and the RT*lap(ln_ps) pressure term
    # blow up AT the poles within a few steps (diagnosed: ps swung to 100-1200 hPa
    # at lat +-90 with zero topography, std ~150 hPa vs obs ~12). Flooring cos_lat
    # at its 80-deg value bounds dx so the handful of pole-most rows integrate with
    # the ~80-deg metric instead of detonating. Accurate cos_lat is kept above for
    # area weighting / Coriolis.
    cos_lat_dyn = jnp.maximum(cos_lat, jnp.cos(jnp.deg2rad(80.0)))
    dx = 2 * jnp.pi * R_EARTH * cos_lat_dyn / nx
    dy = jnp.pi * R_EARTH / ny

    # CFL-safe grid spacing for the diffusion operators ONLY.
    # The dynamical dx above collapses to ~4 m at the pole rows (cos_lat is
    # floored to 1e-5), and the del^4 / del^2 diffusion operators scale as
    # 1/dx^4 and 1/dx^2, so at the poles dt*nu/dx^n exceeds the explicit-Euler
    # stability limit by ~14 orders of magnitude -> overflow / NaN (or, where a
    # safety clamp exists, the pole rows saturate to the clamp every step and
    # inject noise into the meridional gradients). Floor the *diffusion* dx so
    # the few pole-most rows are damped like the first fully-resolved row off the
    # pole (|lat| ~ 80 deg) instead of blowing up. The dynamical dx (gradients,
    # advection, pressure-gradient Laplacian) is left untouched.
    cos_lat_diff = jnp.maximum(jnp.abs(cos_lat), jnp.cos(jnp.deg2rad(80.0)))
    dx_diff = 2 * jnp.pi * R_EARTH * cos_lat_diff / nx

    # 1. Recover U, V from Vorticity, Divergence
    # Pre-filter vorticity/divergence BEFORE Helmholtz inversion.
    # At poles, dx→0 causes kx²→∞ in the wavenumber-dependent coefficients,
    # which amplifies high-frequency noise. The polar filter removes this noise
    # before it gets amplified by the inversion.
    vort_filtered = spectral.polar_filter(state.vorticity, lat_rad)
    div_filtered = spectral.polar_filter(state.divergence, lat_rad)

    # Solve Helmholtz equations: ∇²ψ = ζ, ∇²χ = δ
    psi = spectral.inverse_laplacian(vort_filtered, dx, dy)
    chi = spectral.inverse_laplacian(div_filtered, dx, dy)

    dpsi_dx, dpsi_dy = spectral.compute_gradients(psi, dx, dy)
    dchi_dx, dchi_dy = spectral.compute_gradients(chi, dx, dy)

    u = -dpsi_dy + dchi_dx
    v = dpsi_dx + dchi_dy

    # Clamp winds to prevent supersonic instability in advection. 80 m/s (was 100)
    # is above the strongest real jet-stream cores yet a tighter safety net: with
    # the wind relaxed toward a ~10 m/s climatological jet the field should never
    # approach this, so hitting it flags a developing instability rather than
    # silently saturating at an unphysical 100 m/s.
    u = jnp.clip(u, -80.0, 80.0)
    v = jnp.clip(v, -80.0, 80.0)

    # Filter diagnostic winds to prevent CFL violation near poles
    u = spectral.polar_filter(u, lat_rad)
    v = spectral.polar_filter(v, lat_rad)

    # 2. Physics
    # Precipitation / Convection
    # We use the existing simple physics
    pressure = jnp.exp(state.ln_ps)
    q_sat = physics.compute_saturation_humidity(state.temp, pressure)

    # Use Differentiable Params if provided
    if physics_params is not None and "qc_ref" in physics_params:
        qc_ref = physics_params["qc_ref"]
        epsilon = physics_params.get("epsilon_smooth", physics.EPSILON_SMOOTH)
    else:
        qc_ref = physics.QC_REF
        epsilon = physics.EPSILON_SMOOTH

    precip, heating_precip = physics.compute_precipitation(state.q, q_sat, epsilon=epsilon, qc_ref=qc_ref)
    drying_precip = -precip  # Precip removes moisture (negative tendency)
    # Actually physics.compute_precipitation takes (temp, q). Let's check signature.
    # Assuming it takes (temp, q).

    # Radiation
    # Use state.co2 (spatially varying)
    # We need to average CO2 for the simple radiation scheme or update it to map.
    # The current physics.compute_radiative_forcing takes co2_ppm as float.
    # We will pass the global mean for now or update physics later.
    co2_mean = jnp.mean(state.co2)
    heating_rad = physics.compute_radiative_forcing(
        state.temp, 
        co2_mean, 
        sw_down=sw_down, # Pass seasonal forcing
        lat_rad=lat_rad, 
        solar_constant=solar_constant
    )

    # Momentum relaxation toward a climatological zonal-mean jet.
    #
    # This is a SINGLE-LEVEL (barotropic) atmosphere (ATMOS_GRID.nz=1): with no
    # vertical shear it cannot perform baroclinic instability or thermal-wind
    # balance, so a meridional temperature gradient drives no sustained pressure
    # gradient and the winds geostrophically adjust to rest (diagnosed: |u| decays
    # 10 -> <1 m/s and corr vs ERA5 ~ 0). The eddy-momentum-flux convergence that
    # builds the real surface westerlies, and the Hadley overturning that drives
    # the trades, both live in vertical structure this model does not resolve.
    # We parameterize their net effect as a Rayleigh relaxation toward an observed
    # zonal-mean surface-wind profile (tropical easterlies + mid-latitude
    # westerlies + weak polar easterlies) instead of toward zero.
    #
    # tau = 2 days (not 6): the relaxed jet has meridional shear that is
    # barotropically unstable, and at a 6-day timescale localized eddies grew
    # faster than they were damped (|u| ran to the clamp by ~day 14). A 2-day
    # relaxation damps deviations from the target faster than the instability
    # grows, pinning the field close to the (stable, realistic) climatological jet.
    tau_fric = 86400.0 * 2.0
    abs_lat = jnp.abs(lat[:, None])  # (ny,1) degrees
    u_target = (9.0 * jnp.exp(-((abs_lat - 48.0) / 18.0) ** 2)    # midlat westerlies
                - 5.5 * jnp.exp(-((abs_lat - 13.0) / 11.0) ** 2)  # tropical trades
                - 2.0 * jnp.exp(-((abs_lat - 80.0) / 10.0) ** 2)) # polar easterlies

    # Drag force [m/s2]: relax u toward the climatological jet, v toward zero.
    drag_u = -(u - u_target) / tau_fric
    drag_v = -v / tau_fric

    # Column Mass approx P0/g ~ 1.0e4 kg/m^2
    MASS_COLUMN = 1.0e4

    # Total Forcings
    forcing_u = drag_u
    forcing_v = drag_v

    # Moisture Tendency First
    # q_tendency = drying_precip + flux_latent/Mass
    mass_scaling = 1.0 / MASS_COLUMN
    q_latent_src = flux_latent / 2.5e6 * mass_scaling
    
    q_tendency_raw = drying_precip + q_latent_src
    
    # Clamp Moisture Tendency safely
    # Tighten to 5e-6 kg/kg/s (approx 0.4 g/kg per day? No. 5e-6 * 86400 = 0.4. Correct.)
    # This throttles massive rain shocks.
    forcing_q = jnp.clip(q_tendency_raw, -5e-6, 5e-6)

    # Re-diagnose effective heating from the *actual* amount of precip allowed
    # 1. Limit Precipitation Rate (consistent with forcing_q direction)
    # If forcing_q is negative (drying), max rate is 5e-6.
    precip_limited = jnp.minimum(precip, 1.0e-3) 
    drying_precip_effective = -precip_limited
    
    # 2. Re-compute Heating from LIMITED precip
    heating_precip_effective = (2.5e6 / CP) * precip_limited
    
    # 3. Compute Forcing Q
    forcing_q = drying_precip_effective + q_latent_src
    forcing_q = jnp.clip(forcing_q, -1e-3, 1e-3)
    
    
    # 3. Dynamics Tendencies Helper
    # Advection Operator: -(u ∂/∂x + v ∂/∂y)
    def advect(f):
        df_dx, df_dy = spectral.compute_gradients(f, dx, dy)
        return -(u * df_dx + v * df_dy)

    # 4. Compute Forcing T with EFFECTIVE heating
    # 4. Compute Forcing T with EFFECTIVE heating
    theta_tendency_raw = (heating_precip_effective + heating_rad) + flux_sensible * mass_scaling / CP
    
    # Clamp T Tendency
    # Limit to 0.005 K/s (approx 4.5 K per 15 min step)
    forcing_t = jnp.clip(theta_tendency_raw, -0.005, 0.005)

    forcing_co2 = flux_co2

    # 3. Dynamics Tendencies

    # Coriolis
    f_cor = 2 * OMEGA * sin_lat
    eta = state.vorticity + f_cor

    # Forcing (Friction)
    # Curl(F) for Vorticity, Div(F) for Divergence
    dfu_dx, dfu_dy = spectral.compute_gradients(forcing_u, dx, dy)
    dfv_dx, dfv_dy = spectral.compute_gradients(forcing_v, dx, dy)

    curl_forcing = dfv_dx - dfu_dy
    div_forcing = dfu_dx + dfv_dy

    dzeta_dt = advect(eta) - eta * state.divergence + curl_forcing

    # Divergence Damping + Geostrophic Adjustment
    # Pressure Gradient Term: Div(R * T * Grad(ln_ps))
    dt_dx, dt_dy = spectral.compute_gradients(state.temp, dx, dy)
    dlnps_dx, dlnps_dy = spectral.compute_gradients(state.ln_ps, dx, dy)
    lap_lnps = spectral.compute_laplacian(state.ln_ps, dx, dy)

    term_p = RD * ( (dt_dx * dlnps_dx + dt_dy * dlnps_dy) + state.temp * lap_lnps )

    lap_phi = spectral.compute_laplacian(state.phi_s, dx, dy)

    # Divergence equation. NOTE: the linear divergence-damping term is applied
    # IMPLICITLY below (not here). The previous explicit "-5e-2*div" gave
    # dt*rate = 30*0.05 = 1.5 > 1, so forward Euler overshot and flipped the sign
    # of divergence every step -> a 2dt oscillation that pumped grid-scale noise
    # into ln_ps (which has no smoothing of its own). Divergence damping is the
    # standard device for suppressing spurious gravity-wave / external-mode noise;
    # applied implicitly (below) it is unconditionally stable for any rate.
    ddiv_dt = -term_p - lap_phi + f_cor * state.vorticity + div_forcing

    dt_dt = advect(state.temp) + forcing_t
    dlnps_dt = advect(state.ln_ps) - state.divergence
    dq_dt = advect(state.q) + forcing_q
    dco2_dt = advect(state.co2) + forcing_co2
    # Diffusion: del^4 hyperdiffusion (scale-selective) + polar-only del^2
    # del^4 damps grid-scale noise while preserving large-scale circulation.
    # Polar del^2 handles the lat-lon convergence issue (dx→0 at poles).
    # This replaces the original global del^2 which killed all circulation.

    # del^4 hyperdiffusion coefficients (scale-selective). Kept strong ("tank"
    # values): del^4 damps only the small (k^4) scales and leaves the large scale
    # untouched, so it does NOT fight the climatological jet -- that is now set by
    # the momentum relaxation toward u_target, INDEPENDENT of dissipation strength.
    # This decoupling (relaxation -> jet realism, hyperdiffusion -> stability) is
    # what lets dissipation stay strong: a 4x-weaker setting let a slow eddy
    # instability grow over ~25 days (|u| -> 100 m/s clamp, precip -> 50 mm/day).
    nu4_vort = 3.0e14
    nu4_div = 6.0e14
    nu4_temp = 2.0e13
    nu4_q = 5.0e13
    nu4_co2 = 5.0e13
    # ln_ps had NO scale-selective smoothing of its own (only the zonal polar
    # filter), so grid-scale (2dx) checkerboard noise injected by the divergence
    # accumulated unchecked -> ps_std ~ 150 hPa (obs ~12 hPa). A del^4 term hits
    # only the 2dx noise and leaves the physical (topographic/synoptic) pressure
    # structure intact. Scaled like nu4_temp.
    nu4_lnps = 5.0e12

    # Use the CFL-safe dx_diff (not dx) so the pole rows don't blow up.
    hyperdiff_zeta = spectral.compute_bilaplacian(state.vorticity, dx_diff, dy)
    hyperdiff_div = spectral.compute_bilaplacian(state.divergence, dx_diff, dy)
    hyperdiff_temp = spectral.compute_bilaplacian(state.temp, dx_diff, dy)
    hyperdiff_q = spectral.compute_bilaplacian(state.q, dx_diff, dy)
    hyperdiff_co2 = spectral.compute_bilaplacian(state.co2, dx_diff, dy)
    hyperdiff_lnps = spectral.compute_bilaplacian(state.ln_ps, dx_diff, dy)

    # Polar-only del^2 (active above 60° latitude, zero below)
    nu_base = 2.0e6
    cos_lat_safe = jnp.maximum(jnp.abs(cos_lat), 0.1)
    polar_boost = jnp.clip(1.0 / cos_lat_safe**2, 1.0, 25.0)
    polar_mask = jnp.clip((jnp.abs(lat_rad[:, None]) - jnp.deg2rad(60.0)) / jnp.deg2rad(20.0), 0.0, 1.0)
    nu_polar = nu_base * polar_boost * polar_mask

    diff_zeta = spectral.compute_laplacian(state.vorticity, dx_diff, dy)
    diff_div = spectral.compute_laplacian(state.divergence, dx_diff, dy)
    diff_temp = spectral.compute_laplacian(state.temp, dx_diff, dy)

    # 4. Time Integration (Forward Euler + del^4 + polar del^2)
    new_vorticity = state.vorticity + dt * (dzeta_dt - nu4_vort * hyperdiff_zeta + nu_polar * diff_zeta)
    new_divergence = state.divergence + dt * (ddiv_dt - nu4_div * hyperdiff_div + nu_polar * diff_div)
    # Implicit linear divergence damping: unconditionally stable for any rate (no
    # 2dt overshoot of the old explicit term). 1/DIV_DAMP ~ 100 s -- strong enough
    # to absorb gravity-wave / divergent-mode noise (which otherwise grows into the
    # local |u| blow-up together with the rotational eddies) while still leaving the
    # large-scale divergent (Hadley/Walker) circulation that organises precip.
    DIV_DAMP = 1.0e-2  # [1/s]
    new_divergence = new_divergence / (1.0 + dt * DIV_DAMP)
    new_temp = state.temp + dt * (dt_dt - nu4_temp * hyperdiff_temp + nu_polar * diff_temp)
    new_ln_ps = state.ln_ps + dt * (dlnps_dt - nu4_lnps * hyperdiff_lnps)

    # Clamp Pressure (100 hPa to 1200 hPa)
    new_ln_ps = jnp.clip(new_ln_ps, 9.2, 11.7)
    new_q = state.q + dt * (dq_dt - nu4_q * hyperdiff_q)
    new_co2 = state.co2 + dt * (dco2_dt - nu4_co2 * hyperdiff_co2)

    # Energy Fixer (prevent drift over long runs). alpha reduced 0.1 -> 0.01: with
    # the pole instability removed the scheme is nearly conservative, so only a
    # gentle correction is needed and a strong one would spuriously cool/warm the
    # column in response to transient KE changes.
    new_temp = apply_energy_fixer(
        new_temp, u, v, state.temp, state.u, state.v,
        cos_lat, alpha=0.01
    )

    # 5. Safety Clamping
    new_q = jnp.clip(new_q, 0.0, 0.05)
    new_temp = jnp.clip(new_temp, 150.0, 350.0)
    new_vorticity = jnp.clip(new_vorticity, -5e-4, 5e-4)
    new_divergence = jnp.clip(new_divergence, -5e-4, 5e-4)
    # CO2 was the only prognostic field with no clamp; bound it to keep a stray
    # diffusion/advection overshoot from overflowing to Inf and poisoning the
    # whole field via co2_mean. CO2 is carried in ppm (~280), so the 1e6 ceiling
    # is far above any scenario yet well below float overflow -> never touches
    # physics, only prevents a runaway.
    new_co2 = jnp.clip(new_co2, 0.0, 1.0e6)

    # 6. Polar Filter
    new_vorticity = spectral.polar_filter(new_vorticity, lat_rad)
    new_divergence = spectral.polar_filter(new_divergence, lat_rad)
    new_temp = spectral.polar_filter(new_temp, lat_rad)
    new_ln_ps = spectral.polar_filter(new_ln_ps, lat_rad)
    new_q = spectral.polar_filter(new_q, lat_rad)
    new_co2 = spectral.polar_filter(new_co2, lat_rad)

    new_state = AtmosState(
        vorticity=new_vorticity,
        divergence=new_divergence,
        temp=new_temp,
        ln_ps=new_ln_ps,
        q=new_q,
        co2=new_co2,
        u=u,
        v=v,
        psi=psi,
        chi=chi,
        phi_s=state.phi_s,  # Static topography
    )

    # Return Flux [kg/m^2/s] instead of Tendency [kg/kg/s]
    # precip is [kg/kg/s]. Mass Column ~ 1e4 kg/m^2.
    precip_flux = precip * 1.0e4 
    
    return new_state, (precip_flux, jnp.exp(new_ln_ps))
