"""Multi-level atmosphere on Google's `dinosaur` spectral primitive-equation dycore.

Replaces the single-level (barotropic) atmosphere, which could not perform
baroclinic instability (no synoptic systems, no real ITCZ). This wraps dinosaur's
differentiable dry dycore (T31, sigma levels, ImEx-RK3) with:

* an SST-COUPLED thermal forcing -- a Held-Suarez-style Newtonian relaxation whose
  radiative-equilibrium temperature is anchored to the underlying SST at the
  surface (instead of HS's idealized 315 - 60 sin^2(lat)), keeping HS's vertical
  lapse structure, boundary-layer relaxation rates, and Rayleigh drag; and
* a PROGNOSTIC MOISTURE cycle: specific humidity is carried as a tracer (advected
  by the dycore), with a surface evaporation source (bulk formula from SST) and a
  large-scale condensation sink that rains out supersaturation and latent-heats
  the column. Moisture converging in the tropics -> condensation -> an ITCZ.

So the atmosphere RESPONDS to the ocean and has a real hydrological cycle.

Design notes
------------
* The dycore is `dinosaur.primitive_equations.PrimitiveEquations` (implicit-
  explicit); it already advects tracers. The SST + moisture physics are extra
  EXPLICIT tendencies summed into it.
* SST is constant within one ocean coupling interval, so `run_interval(state,
  sst, n_steps)` is jitted ONCE with `sst` as a traced argument.
* Unit handling: all physics is done in plain jax with float scale factors
  precomputed in __init__ (no pint inside jit). Temperature scale is 1 K, so
  numeric temperatures are Kelvin. SI rates [1/s] -> nondim by * self.t0_seconds.
* `jcm` is intentionally NOT used (its PyPI build is broken); see CHANGELOG.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from dinosaur import (coordinate_systems, spherical_harmonic, sigma_coordinates,
                      primitive_equations as pe, primitive_equations_states as pes,
                      time_integration as ti, scales, xarray_utils)

units = scales.units

# Thermodynamic constants (SI).
RHO_AIR = 1.2        # kg/m^3
CE = 1.2e-3          # moisture exchange coefficient
G = 9.81             # m/s^2
LV = 2.5e6           # J/kg latent heat of vaporization
CP = 1004.0          # J/kg/K
EPS = 0.622          # Rd/Rv


def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def save_state(state, path):
    """Persist a dinosaur modal State (complex spectral coefficients + humidity
    tracer + sim_time) to a .npz for restarting long coupled runs."""
    np.savez(path,
             vorticity=np.asarray(state.vorticity),
             divergence=np.asarray(state.divergence),
             temperature_variation=np.asarray(state.temperature_variation),
             log_surface_pressure=np.asarray(state.log_surface_pressure),
             specific_humidity=np.asarray(state.tracers["specific_humidity"]),
             sim_time=np.asarray(float(state.sim_time) if state.sim_time is not None else 0.0))


def load_state(path):
    """Load a dinosaur modal State saved by `save_state`.

    NOTE: ``sim_time`` is restored as None (the dycore's convention for an unset
    clock), NOT as the saved scalar. The ImEx integrator forms tendencies with
    sim_time=None and adds them to the state via tree_map; a *scalar* sim_time on
    the state would mismatch (array + None) and break the step. Absolute run time
    is tracked by the caller (the run harness counts days), so nothing is lost.
    The saved sim_time remains in the .npz for reference.
    """
    d = np.load(path, allow_pickle=True)
    return pe.State(
        vorticity=jnp.asarray(d["vorticity"]),
        divergence=jnp.asarray(d["divergence"]),
        temperature_variation=jnp.asarray(d["temperature_variation"]),
        log_surface_pressure=jnp.asarray(d["log_surface_pressure"]),
        tracers={"specific_humidity": jnp.asarray(d["specific_humidity"])},
        sim_time=None)


def _qsat(temp_K, p_Pa):
    """Saturation specific humidity (kg/kg) via Clausius-Clapeyron."""
    t_c = temp_K - 273.15
    e_sat = 611.0 * jnp.exp(17.27 * t_c / (t_c + 237.3))
    return EPS * e_sat / jnp.maximum(p_Pa, 1.0)


class DinoAtmosphere:
    """SST-coupled moist multi-level atmosphere (dinosaur dycore + physics)."""

    def __init__(self, truncation="T31", layers=24, dt_minutes=20.0,
                 diffusion_tau_hours=6.0, diffusion_order=2,
                 sigma_b=0.7, kf_per_day=1.0, ka_per_day=1 / 40.0, ks_per_day=1 / 4.0,
                 minT=200.0, dThz=10.0, tau_cond_hours=3.0, init_rh=0.5,
                 orography=True, seed_wind_ms=2.0, seed=0):
        grid = getattr(spherical_harmonic.Grid, truncation)()
        self.coords = coordinate_systems.CoordinateSystem(
            horizontal=grid,
            vertical=sigma_coordinates.SigmaCoordinates.equidistant(layers))
        self.specs = pe.PrimitiveEquationsSpecs.from_si()
        self.layers = layers

        # grid geometry (needed before the initial state to place orography)
        _, sin_lat = grid.nodal_mesh                                 # (nlon, nlat)
        self.sin_lat = np.asarray(sin_lat)
        self.cos_lat = np.sqrt(np.maximum(1 - self.sin_lat ** 2, 1e-12))
        self.lat_deg = np.rad2deg(np.arcsin(self.sin_lat[0]))        # (nlat,)
        self.nlon, self.nlat = self.sin_lat.shape

        # ETOPO orography on the dinosaur grid. data.load_topography returns
        # elevation [m] (>=0, ocean=0) on a linear-lat grid; interpolate onto the
        # Gaussian latitudes, index-preserving in longitude (the SAME convention
        # the coupled driver uses to regrid SST in), so SST and orography stay
        # mutually aligned on the dinosaur grid. Real topography gives the
        # stationary waves and the topographically-shaped surface pressure that
        # an aquaplanet cannot have.
        self.orography_m = np.zeros((self.nlon, self.nlat))
        if orography:
            from chronos_esm import data
            from scipy.ndimage import gaussian_filter
            topo_lin = np.asarray(data.load_topography(self.nlat, self.nlon))  # (nlat,nlon)
            lat_lin = np.linspace(-90, 90, self.nlat)
            topo_g = np.stack([np.interp(self.lat_deg, lat_lin, topo_lin[:, j])
                               for j in range(self.nlon)], axis=0)             # (nlon,nlat)
            # Smooth the orography before use. Raw ETOPO at T31 has sharp gradients
            # that produce Gibbs ripples in the spectral (to_modal) representation;
            # those ripples drive a spurious mass leak (surface pressure drifted
            # ~1.8 hPa/day). A mild Gaussian smooth (wrap in lon, reflect in lat)
            # removes the sub-grid sharpness and stabilizes the global-mean pressure.
            self.orography_m = np.maximum(
                gaussian_filter(topo_g, sigma=(1.2, 1.2), mode=("wrap", "reflect")), 0.0)

        init_fn, aux = pes.isothermal_rest_atmosphere(
            self.coords, self.specs, p0=1e5 * units.pascal, p1=5e3 * units.pascal,
            surface_height=jnp.asarray(self.orography_m) * units.meter)
        self.ref_temps = aux[xarray_utils.REF_TEMP_KEY]            # (layers,) ~288 K
        self.orography_modal = self.coords.horizontal.to_modal(aux[xarray_utils.OROGRAPHY])
        base_state = init_fn(rng_key=jax.random.PRNGKey(0))

        self.eq = pe.PrimitiveEquations(
            self.ref_temps, self.orography_modal, self.coords, self.specs)

        # nondimensional constants (temperature scale is 1 K, so K == nondim T)
        self.dt = self.specs.nondimensionalize(dt_minutes * units.minute)
        self.steps_per_day = int(round(self.specs.nondimensionalize(1 * units.day) / self.dt))
        # Horizontal hyperdiffusion. tau is the e-folding time of the TOP (grid-
        # scale) mode; mode l is damped as (l(l+1)/L(L+1))^order, so tau still
        # controls grid noise fast (~tau) while leaving large scales nearly free.
        # tau=6 h (NOT the very strong tau~2 h): at coarse T31 the baroclinic
        # eddies live at fairly high total wavenumbers (l~10-15) where tau=2 h
        # already damps them on 1-15 days -- competitive with their ~1-3 day
        # baroclinic growth -- which SUPPRESSES the eddies and hence the eddy
        # momentum flux that drives the mid-latitude surface westerlies. tau=6 h
        # lets those eddies grow (storm tracks, surface westerlies) while still
        # killing grid-scale noise. (This is the OPPOSITE lesson from the legacy
        # single-level model, where eddies were spurious and diffusion was kept
        # strong to suppress them; here eddies are the physics we want.)
        self.diff = ti.horizontal_diffusion_step_filter(
            grid, self.dt,
            tau=self.specs.nondimensionalize(diffusion_tau_hours * units.hour),
            order=diffusion_order)
        self.p0 = self.specs.nondimensionalize(1e5 * units.pascal)
        self.kf = self.specs.nondimensionalize(kf_per_day / units.day)
        self.ka = self.specs.nondimensionalize(ka_per_day / units.day)
        self.ks = self.specs.nondimensionalize(ks_per_day / units.day)
        self.kappa = self.specs.kappa
        self.sigma_b = sigma_b
        self.minT = self.specs.nondimensionalize(minT * units.degK)
        self.dThz = self.specs.nondimensionalize(dThz * units.degK)
        self.tau_cond = self.specs.nondimensionalize(tau_cond_hours * units.hour)
        # unit scale factors (plain floats, jit-safe)
        self.p_scale_Pa = float(self.specs.dimensionalize(1.0, units.pascal).magnitude)
        self.v_scale_ms = float(self.specs.dimensionalize(1.0, units.meter / units.second).magnitude)
        self.t0_seconds = float(self.specs.dimensionalize(1.0, units.second).magnitude)  # s per nondim time

        # vertical grid
        self.sigma = np.asarray(self.coords.vertical.centers)        # (layers,)
        self.dsigma = 1.0 / layers                                   # equidistant layer thickness

        # HS vertical profiles of relaxation/drag (layers,1,1)
        cutoff = np.maximum(0.0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
        self._kv = (self.kf * cutoff)[:, None, None]
        self._kt = self.ka + (self.ks - self.ka) * (cutoff[:, None, None] * self.cos_lat ** 4)

        # initial state with a humidity tracer (init_rh * q_sat at the rest state)
        nodal_sp = jnp.exp(self.coords.horizontal.to_nodal(base_state.log_surface_pressure))
        p_lvl = self.sigma[:, None, None] * nodal_sp * self.p_scale_Pa
        q0_nodal = init_rh * _qsat(self.ref_temps[:, None, None], p_lvl)
        q0_modal = self.coords.horizontal.to_modal(q0_nodal)

        # Seed a small random ROTATIONAL (vorticity) perturbation to BREAK ZONAL
        # SYMMETRY. isothermal_rest_atmosphere is EXACTLY axisymmetric (vorticity
        # == divergence == temperature_variation == 0), an unstable equilibrium the
        # flow never leaves on its own (only floating-point roundoff breaks it,
        # after ~100 days). Without an asymmetric seed NO baroclinic eddies form,
        # so there is no eddy momentum-flux convergence and NO mid-latitude surface
        # westerlies (the surface stays easterly everywhere -> u_sfc pattern corr
        # ~0). A *velocity* seed is used, not a temperature one: a temperature
        # perturbation is simply relaxed away by the thermal forcing (_kt) before
        # it can grow, whereas a rotational velocity seed projects directly onto
        # the growing baroclinic mode and reliably triggers eddies + storm tracks.
        seed_vort = jnp.zeros_like(base_state.vorticity)
        if seed_wind_ms > 0:
            k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
            shp = (self.layers,) + self.coords.horizontal.nodal_shape
            amp = seed_wind_ms / self.v_scale_ms
            cl = jnp.asarray(self.cos_lat)[None]
            clu = jnp.stack([cl * amp * jax.random.normal(k1, shp),
                             cl * amp * jax.random.normal(k2, shp)])
            seed_vort = self.coords.horizontal.curl_cos_lat(self.coords.horizontal.to_modal(clu))

        # base state = isothermal rest + humidity tracer + the rotational seed.
        # initial_state() adds the near-equilibrium temperature on top (see there).
        self._base_state = dataclasses.replace(
            base_state, vorticity=base_state.vorticity + seed_vort,
            tracers={"specific_humidity": q0_modal})

        # Generic Earth-like SST gradient for the default near-equilibrium init
        # (used when initial_state() is called without a specific SST).
        default_sst_K = 300.0 - 40.0 * self.sin_lat ** 2
        self._default_sst_nd = self.specs.nondimensionalize(
            jnp.asarray(default_sst_K) * units.degK)

        # dry-mass fixer target: the initial area-weighted global-mean surface
        # pressure. Restoring it each interval guarantees conservation of global
        # dry mass (orography otherwise leaks ~0.3 hPa/day even after smoothing).
        self._area_w = jnp.asarray(self.cos_lat)[None]   # (1, nlon, nlat)
        init_ps = jnp.exp(self.coords.horizontal.to_nodal(base_state.log_surface_pressure))
        self._target_ps_mean = float(jnp.sum(init_ps * self._area_w) / jnp.sum(self._area_w))

        self._run = jax.jit(self._run_interval, static_argnums=(2,))

    def _fix_mass(self, state):
        """Rescale surface pressure to conserve the initial global-mean dry mass."""
        ps = jnp.exp(self.coords.horizontal.to_nodal(state.log_surface_pressure))
        cur = jnp.sum(ps * self._area_w) / jnp.sum(self._area_w)
        lnps = self.coords.horizontal.to_modal(jnp.log(ps * (self._target_ps_mean / cur)))
        return dataclasses.replace(state, log_surface_pressure=lnps)

    # ---- SST-anchored equilibrium temperature ----
    def _equilibrium_temperature(self, nodal_surface_pressure, sst_nd):
        p_over_p0 = self.sigma[:, None, None] * nodal_surface_pressure / self.p0
        temperature = p_over_p0 ** self.kappa * (
            sst_nd[None, :, :] - self.dThz * jnp.log(p_over_p0) * self.cos_lat ** 2)
        return jnp.maximum(self.minT, temperature)

    # ---- combined explicit forcing: HS relaxation + drag + moisture ----
    def _forcing_explicit(self, state, sst_nd):
        aux = pe.compute_diagnostic_state(state=state, coords=self.coords)

        # Rayleigh drag (HS kv).
        nodal_velocity_tendency = jax.tree.map(
            lambda x: -self._kv * x / self.coords.horizontal.cos_lat ** 2, aux.cos_lat_u)

        # Thermal relaxation toward the SST-anchored equilibrium.
        nodal_T = self.ref_temps[:, None, None] + aux.temperature_variation
        nodal_sp = jnp.exp(self.coords.horizontal.to_nodal(state.log_surface_pressure))
        Teq = self._equilibrium_temperature(nodal_sp, sst_nd)
        nodal_T_tend = -self._kt * (nodal_T - Teq)

        # --- moisture physics (nodal) ---
        psurf_Pa = nodal_sp[0] * self.p_scale_Pa            # (nlon, nlat)
        p_lvl_Pa = self.sigma[:, None, None] * psurf_Pa     # (layers, nlon, nlat)
        q = self.coords.horizontal.to_nodal(state.tracers["specific_humidity"])
        q = jnp.maximum(q, 0.0)

        # surface evaporation into the bottom level (bulk formula from SST)
        u = aux.cos_lat_u[0] / self.coords.horizontal.cos_lat
        v = aux.cos_lat_u[1] / self.coords.horizontal.cos_lat
        wind_ms = jnp.clip(jnp.sqrt(u[-1] ** 2 + v[-1] ** 2) * self.v_scale_ms, 1.0, 25.0)
        qsat_sst = _qsat(sst_nd, psurf_Pa)
        evap_si = RHO_AIR * CE * wind_ms * jnp.maximum(qsat_sst - q[-1], 0.0)  # kg/m^2/s
        dq_dt_evap = evap_si * G / (psurf_Pa * self.dsigma) * self.t0_seconds   # nondim 1/t
        dq = jnp.zeros_like(q).at[-1].add(dq_dt_evap)

        # large-scale condensation: rain out supersaturation, latent-heat the column
        qsat = _qsat(nodal_T, p_lvl_Pa)
        cond = jnp.maximum(q - qsat, 0.0) / self.tau_cond                       # nondim 1/t
        dq = dq - cond
        nodal_T_tend = nodal_T_tend + (LV / CP) * cond

        # to modal
        return pe.State(
            vorticity=self.coords.horizontal.curl_cos_lat(
                self.coords.horizontal.to_modal(nodal_velocity_tendency)),
            divergence=self.coords.horizontal.div_cos_lat(
                self.coords.horizontal.to_modal(nodal_velocity_tendency)),
            temperature_variation=self.coords.horizontal.to_modal(nodal_T_tend),
            log_surface_pressure=jnp.zeros_like(state.log_surface_pressure),
            tracers={"specific_humidity": self.coords.horizontal.to_modal(dq)})

    def _run_interval(self, state, sst_nd, n_steps):
        def explicit(s):
            return _tree_add(self.eq.explicit_terms(s), self._forcing_explicit(s, sst_nd))
        ode = ti.ImplicitExplicitODE.from_functions(
            explicit, self.eq.implicit_terms, self.eq.implicit_inverse)
        step = ti.step_with_filters(ti.imex_rk_sil3(ode, self.dt), [self.diff])
        return self._fix_mass(ti.repeated(step, n_steps)(state))

    # ---- public API ----
    def initial_state(self, sst_nodal_K=None):
        """Initial modal state, primed to spin up baroclinic eddies quickly.

        Temperature is initialized near RADIATIVE EQUILIBRIUM (the SST-anchored
        Held-Suarez Teq) instead of isothermal rest, so the equator-pole gradient
        -- and hence the baroclinic jet -- exists from day 0 rather than taking
        ~2 months to build through the slow (40-day) thermal relaxation. Combined
        with the rotational seed baked into the base state, eddies and mid-latitude
        surface westerlies appear within ~2-3 weeks instead of ~2-3 months.

        Pass the SST (nodal, K, shape (nlon, nlat)) to tailor the equilibrium to
        the run's SST; with no argument a generic Earth-like gradient is used and
        the run then relaxes to whatever SST is supplied to step().
        """
        if sst_nodal_K is None:
            sst_nd = self._default_sst_nd
        else:
            sst_nd = self.specs.nondimensionalize(jnp.asarray(sst_nodal_K) * units.degK)
        nodal_sp = jnp.exp(self.coords.horizontal.to_nodal(self._base_state.log_surface_pressure))
        Teq = self._equilibrium_temperature(nodal_sp, sst_nd)
        tvar = self.coords.horizontal.to_modal(Teq - self.ref_temps[:, None, None])
        return dataclasses.replace(self._base_state, temperature_variation=tvar)

    def step(self, state, sst_nodal_K, n_days=1):
        """Advance n_days with SST [K] on the dinosaur nodal grid (nlon, nlat)."""
        sst_nd = self.specs.nondimensionalize(jnp.asarray(sst_nodal_K) * units.degK)
        return self._run(state, sst_nd, int(self.steps_per_day * n_days))

    def diagnostics(self, state):
        """Dimensional fields on the dinosaur nodal grid (nlon, nlat).

        Keys: u, v [m/s] (layers,..); temperature [K]; specific_humidity [kg/kg];
        surface u_sfc/v_sfc/t_sfc/q_sfc; surface_pressure [Pa]; precip [kg/m^2/s];
        lat_deg.
        """
        d = pe.compute_diagnostic_state(state=state, coords=self.coords)
        clu = np.asarray(d.cos_lat_u)
        u = self.specs.dimensionalize(clu[0] / self.cos_lat[None], units.meter / units.second).magnitude
        v = self.specs.dimensionalize(clu[1] / self.cos_lat[None], units.meter / units.second).magnitude
        T = self.specs.dimensionalize(
            self.ref_temps[:, None, None] + np.asarray(d.temperature_variation), units.degK).magnitude
        nodal_sp = np.exp(np.asarray(self.coords.horizontal.to_nodal(state.log_surface_pressure)))
        ps_Pa = nodal_sp[0] * self.p_scale_Pa               # (nlon, nlat)
        q = np.asarray(self.coords.horizontal.to_nodal(state.tracers["specific_humidity"]))
        q = np.maximum(q, 0.0)
        # column precipitation = vertical integral of condensation [kg/m^2/s]
        p_lvl = self.sigma[:, None, None] * ps_Pa
        qsat = np.asarray(_qsat(jnp.asarray(T), jnp.asarray(p_lvl)))
        cond_si = np.maximum(q - qsat, 0.0) / (self.tau_cond * self.t0_seconds)  # 1/s
        precip = np.sum(cond_si * (ps_Pa * self.dsigma) / G, axis=0)            # kg/m^2/s (nlon,nlat)
        # reduce surface pressure to mean sea level with a fixed standard-atmosphere
        # scale height (static, noise-free) so it is comparable to ERA5 MSL.
        mslp = ps_Pa * np.exp(G * self.orography_m / (287.0 * 288.0))           # Pa
        return dict(u=u, v=v, temperature=T, specific_humidity=q,
                    u_sfc=u[-1], v_sfc=v[-1], t_sfc=T[-1], q_sfc=q[-1],
                    surface_pressure=ps_Pa, mslp=mslp, precip=precip, lat_deg=self.lat_deg)
