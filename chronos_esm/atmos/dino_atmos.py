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


def _qsat(temp_K, p_Pa):
    """Saturation specific humidity (kg/kg) via Clausius-Clapeyron."""
    t_c = temp_K - 273.15
    e_sat = 611.0 * jnp.exp(17.27 * t_c / (t_c + 237.3))
    return EPS * e_sat / jnp.maximum(p_Pa, 1.0)


class DinoAtmosphere:
    """SST-coupled moist multi-level atmosphere (dinosaur dycore + physics)."""

    def __init__(self, truncation="T31", layers=24, dt_minutes=20.0,
                 diffusion_tau_hours=2.0, diffusion_order=2,
                 sigma_b=0.7, kf_per_day=1.0, ka_per_day=1 / 40.0, ks_per_day=1 / 4.0,
                 minT=200.0, dThz=10.0, tau_cond_hours=3.0, init_rh=0.5):
        grid = getattr(spherical_harmonic.Grid, truncation)()
        self.coords = coordinate_systems.CoordinateSystem(
            horizontal=grid,
            vertical=sigma_coordinates.SigmaCoordinates.equidistant(layers))
        self.specs = pe.PrimitiveEquationsSpecs.from_si()
        self.layers = layers

        init_fn, aux = pes.isothermal_rest_atmosphere(
            self.coords, self.specs, p0=1e5 * units.pascal, p1=5e3 * units.pascal)
        self.ref_temps = aux[xarray_utils.REF_TEMP_KEY]            # (layers,) ~288 K
        self.orography_modal = self.coords.horizontal.to_modal(aux[xarray_utils.OROGRAPHY])
        base_state = init_fn(rng_key=jax.random.PRNGKey(0))

        self.eq = pe.PrimitiveEquations(
            self.ref_temps, self.orography_modal, self.coords, self.specs)

        # nondimensional constants (temperature scale is 1 K, so K == nondim T)
        self.dt = self.specs.nondimensionalize(dt_minutes * units.minute)
        self.steps_per_day = int(round(self.specs.nondimensionalize(1 * units.day) / self.dt))
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

        # grid geometry
        self.sigma = np.asarray(self.coords.vertical.centers)        # (layers,)
        self.dsigma = 1.0 / layers                                   # equidistant layer thickness
        _, sin_lat = grid.nodal_mesh                                 # (nlon, nlat)
        self.sin_lat = np.asarray(sin_lat)
        self.cos_lat = np.sqrt(np.maximum(1 - self.sin_lat ** 2, 1e-12))
        self.lat_deg = np.rad2deg(np.arcsin(self.sin_lat[0]))        # (nlat,)
        self.nlon, self.nlat = self.sin_lat.shape

        # HS vertical profiles of relaxation/drag (layers,1,1)
        cutoff = np.maximum(0.0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
        self._kv = (self.kf * cutoff)[:, None, None]
        self._kt = self.ka + (self.ks - self.ka) * (cutoff[:, None, None] * self.cos_lat ** 4)

        # initial state with a humidity tracer (init_rh * q_sat at the rest state)
        nodal_sp = jnp.exp(self.coords.horizontal.to_nodal(base_state.log_surface_pressure))
        p_lvl = self.sigma[:, None, None] * nodal_sp * self.p_scale_Pa
        q0_nodal = init_rh * _qsat(self.ref_temps[:, None, None], p_lvl)
        q0_modal = self.coords.horizontal.to_modal(q0_nodal)
        self._init_state = dataclasses.replace(
            base_state, tracers={"specific_humidity": q0_modal})

        self._run = jax.jit(self._run_interval, static_argnums=(2,))

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
        return ti.repeated(step, n_steps)(state)

    # ---- public API ----
    def initial_state(self):
        return self._init_state

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
        return dict(u=u, v=v, temperature=T, specific_humidity=q,
                    u_sfc=u[-1], v_sfc=v[-1], t_sfc=T[-1], q_sfc=q[-1],
                    surface_pressure=ps_Pa, precip=precip, lat_deg=self.lat_deg)
