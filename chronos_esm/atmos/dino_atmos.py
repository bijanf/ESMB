"""Multi-level atmosphere on Google's `dinosaur` spectral primitive-equation dycore.

Replaces the single-level (barotropic) atmosphere, which could not perform
baroclinic instability (no synoptic systems, no real ITCZ). This wraps dinosaur's
differentiable dry dycore (T31, sigma levels, ImEx-RK3) with an SST-COUPLED
thermal forcing: a Held-Suarez-style Newtonian relaxation whose radiative-
equilibrium temperature is anchored to the underlying sea-surface temperature at
the surface (instead of HS's idealized 315 - 60 sin^2(lat) profile), keeping HS's
vertical lapse structure, boundary-layer relaxation rates, and Rayleigh drag.

So the atmosphere RESPONDS to the ocean: a realistic SST gradient drives a
realistic temperature gradient -> baroclinic jets + trades via genuine dynamics.

Design notes
------------
* The dycore is `dinosaur.primitive_equations.PrimitiveEquations` (implicit-
  explicit). The SST forcing is an extra EXPLICIT tendency summed into it.
* SST is constant within one ocean coupling interval, so `run_interval(state,
  sst, n_steps)` is jitted ONCE with `sst` as a traced argument and reused every
  interval (no re-tracing as the ocean evolves).
* State is modal (vorticity / divergence / temperature_variation /
  log_surface_pressure). Surface (10 m-ish) fields are the lowest sigma level.
* `jcm` is intentionally NOT used (its PyPI build is broken); we build on
  dinosaur directly. See experiments/dino_held_suarez.py and CHANGELOG 2026-06-11d.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from dinosaur import (coordinate_systems, spherical_harmonic, sigma_coordinates,
                      primitive_equations as pe, primitive_equations_states as pes,
                      time_integration as ti, scales, xarray_utils)

units = scales.units


def _tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


class DinoAtmosphere:
    """SST-coupled multi-level atmosphere (dinosaur dycore + HS-style forcing)."""

    def __init__(self, truncation="T31", layers=24, dt_minutes=20.0,
                 diffusion_tau_hours=2.0, diffusion_order=2,
                 sigma_b=0.7, kf_per_day=1.0, ka_per_day=1 / 40.0, ks_per_day=1 / 4.0,
                 minT=200.0, dThz=10.0):
        grid = getattr(spherical_harmonic.Grid, truncation)()
        self.coords = coordinate_systems.CoordinateSystem(
            horizontal=grid,
            vertical=sigma_coordinates.SigmaCoordinates.equidistant(layers))
        self.specs = pe.PrimitiveEquationsSpecs.from_si()
        self.layers = layers

        init_fn, aux = pes.isothermal_rest_atmosphere(
            self.coords, self.specs, p0=1e5 * units.pascal, p1=5e3 * units.pascal)
        self.ref_temps = aux[xarray_utils.REF_TEMP_KEY]            # (layers,) ~288
        self.orography_modal = self.coords.horizontal.to_modal(aux[xarray_utils.OROGRAPHY])
        self._init_state = init_fn(rng_key=jax.random.PRNGKey(0))

        self.eq = pe.PrimitiveEquations(
            self.ref_temps, self.orography_modal, self.coords, self.specs)

        # --- nondimensional constants (temperature scale is 1 K, so K == nondim) ---
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

        # --- grid geometry ---
        self.sigma = np.asarray(self.coords.vertical.centers)        # (layers,)
        _, sin_lat = grid.nodal_mesh                                 # (nlon, nlat)
        self.sin_lat = np.asarray(sin_lat)
        self.cos_lat = np.sqrt(np.maximum(1 - self.sin_lat ** 2, 1e-12))
        self.lat_deg = np.rad2deg(np.arcsin(self.sin_lat[0]))        # (nlat,)
        self.nlon, self.nlat = self.sin_lat.shape

        # HS vertical profiles of relaxation/drag (layers,1,1)
        cutoff = np.maximum(0.0, (self.sigma - self.sigma_b) / (1 - self.sigma_b))
        self._kv = (self.kf * cutoff)[:, None, None]
        self._kt = self.ka + (self.ks - self.ka) * (cutoff[:, None, None] * self.cos_lat ** 4)

        self._run = jax.jit(self._run_interval, static_argnums=(2,))

    # ---- forcing: HS dynamics relaxation with SST-anchored Teq ----
    def _equilibrium_temperature(self, nodal_surface_pressure, sst_nd):
        """Teq(sigma, lat, lon): anchored to SST at the surface, HS lapse aloft."""
        p_over_p0 = self.sigma[:, None, None] * nodal_surface_pressure / self.p0
        temperature = p_over_p0 ** self.kappa * (
            sst_nd[None, :, :] - self.dThz * jnp.log(p_over_p0) * self.cos_lat ** 2)
        return jnp.maximum(self.minT, temperature)

    def _forcing_explicit(self, state, sst_nd):
        aux = pe.compute_diagnostic_state(state=state, coords=self.coords)
        # Rayleigh drag in the boundary layer (HS kv).
        nodal_velocity_tendency = jax.tree.map(
            lambda x: -self._kv * x / self.coords.horizontal.cos_lat ** 2, aux.cos_lat_u)
        # Thermal relaxation toward the SST-anchored equilibrium.
        nodal_temperature = self.ref_temps[:, None, None] + aux.temperature_variation
        nodal_sp = jnp.exp(self.coords.horizontal.to_nodal(state.log_surface_pressure))
        Teq = self._equilibrium_temperature(nodal_sp, sst_nd)
        nodal_temperature_tendency = -self._kt * (nodal_temperature - Teq)
        # to modal
        temperature_tendency = self.coords.horizontal.to_modal(nodal_temperature_tendency)
        velocity_tendency = self.coords.horizontal.to_modal(nodal_velocity_tendency)
        return pe.State(
            vorticity=self.coords.horizontal.curl_cos_lat(velocity_tendency),
            divergence=self.coords.horizontal.div_cos_lat(velocity_tendency),
            temperature_variation=temperature_tendency,
            log_surface_pressure=jnp.zeros_like(state.log_surface_pressure))

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
        """Dimensional fields on the dinosaur nodal grid.

        Returns dict: u, v [m/s] (layers,nlon,nlat); temperature [K]; surface u/v/T
        (lowest level); surface_pressure [Pa]; lat_deg.
        """
        d = pe.compute_diagnostic_state(state=state, coords=self.coords)
        clu = np.asarray(d.cos_lat_u)                               # (2,layers,nlon,nlat)
        u = self.specs.dimensionalize(clu[0] / self.cos_lat[None], units.meter / units.second).magnitude
        v = self.specs.dimensionalize(clu[1] / self.cos_lat[None], units.meter / units.second).magnitude
        T = self.specs.dimensionalize(
            self.ref_temps[:, None, None] + np.asarray(d.temperature_variation),
            units.degK).magnitude
        nodal_sp = np.exp(np.asarray(self.coords.horizontal.to_nodal(state.log_surface_pressure)))
        ps = self.specs.dimensionalize(nodal_sp, units.pascal).magnitude  # (1,nlon,nlat)
        return dict(u=u, v=v, temperature=T,
                    u_sfc=u[-1], v_sfc=v[-1], t_sfc=T[-1], surface_pressure=ps[0],
                    lat_deg=self.lat_deg)
