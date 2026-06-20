"""Differentiable multi-level (dinosaur) coupled stepper — working-ESM P1.

Promotes the standalone ``experiments/run_dino_coupled.py`` loop body into the
library as ONE differentiable, jittable coupling-interval step, with the dinosaur
atmosphere as THE active model (the owner's non-negotiable). The legacy single-level
path in ``main.step_coupled`` is untouched — this is the ``atmos_backend='dino'`` model.

The coupled state carries the dinosaur modal state (``dinosaur.primitive_equations.
State`` — a registered JAX pytree) DIRECTLY alongside the ocean and land states, so
the whole thing flows through ``jax.jit`` / ``jax.grad`` / ``jax.lax.scan``. All the
graph-cut numpy helpers are replaced by their jnp ports in
``chronos_esm.coupler.dino_coupling``.

One coupling interval:
    SST(lin) -> surf_T over land -> regrid -> DinoAtmosphere interval (remat'd scan)
    -> jnp diagnostics -> regrid -> land step + bulk ocean fluxes -> ocean scan (remat'd)

Sea ice (the Semtner thermodynamic model in ``chronos_esm/ice``) is NOT yet wired in
here (the standalone harness this promotes also had none); that is the next P1 chunk.

See memory ``working-esm-roadmap-2026-06-19`` (P1) / workflow wxf3z5fpu.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from dinosaur import primitive_equations as pe

from chronos_esm import main
from chronos_esm.atmos import physics as aphys
from chronos_esm.atmos.dino_atmos import (DinoAtmosphere, save_state as dino_save,
                                          load_state as dino_load)
from chronos_esm.config import (OCEAN_DZ, OCEAN_GRID, EARTH_RADIUS, DT_OCEAN,
                                DRAG_COEFF_LAND)
from chronos_esm.land.driver import LandState, step_land
from chronos_esm.ice.driver import IceState, init_ice_state, step_ice
from chronos_esm.ocean import veros_driver
from chronos_esm.ocean.veros_driver import OceanState
from chronos_esm.coupler import dino_coupling as dc


class DinoCoupledState(NamedTuple):
    """Full state of the multi-level coupled model. ``atmos`` is the dinosaur modal
    ``pe.State`` (spectral coefficients + humidity tracer), carried as a pytree so
    the whole state differentiates. ``day`` is the absolute simulation day."""
    ocean: object        # chronos_esm.ocean.veros_driver.OceanState
    atmos: pe.State      # dinosaur modal state
    land: object         # chronos_esm.land.driver.LandState
    ice: IceState        # Semtner 0-layer thermodynamic sea ice
    day: float


class DinoCoupledModel:
    """Differentiable dinosaur<->ocean<->land coupled model (T31 atmos / ocean grid).

    The atmosphere object and all grid metrics/masks are static (held on the
    instance, closed over by the jitted step), so the only traced data is the
    ``DinoCoupledState`` pytree.
    """

    def __init__(self, ocean_ic="woa", restore_to_woa=True, restore_tau_days=30.0,
                 q_flux=None, interval=1.0, thc_haline_gain=1.0,
                 thc_contrast_depth_m=None):
        """restore_tau_days/q_flux select the SST flux-correction mode:
          - q_flux=None, tau~30  -> strong Haney restoring to WOA (CONTROL mode);
          - q_flux=<field>, tau long (e.g. 3650) -> frozen q-flux + weak anomaly
            restoring (FREE / forcing-responsive mode; see ocean.flux_correction)."""
        base = main.init_model(ocean_ic=ocean_ic)
        self._ocean0 = base.ocean
        self._land0 = base.land
        # WOA SST target for the flux-correction (captured from the WOA init).
        self.sst_target = jnp.asarray(base.ocean.temp[0]) if restore_to_woa else None
        self.restore_tau_days = restore_tau_days
        self.q_flux = None if q_flux is None else jnp.asarray(q_flux)
        self.thc_haline_gain = thc_haline_gain   # >1 -> salt-advection feedback for tipping
        # convection-layer depth for the density contrast; shallow (~300 m) gives a
        # surface hosing leverage on deep-water formation (None -> use overturning depth).
        self.thc_contrast_depth_m = thc_contrast_depth_m

        nz = base.ocean.u.shape[0]
        self.nz = nz
        self.ocean_mask_3d, self.surface_mask = main.ocean_masks(nz=nz)
        self.omask = jnp.asarray(np.asarray(self.surface_mask).astype(bool))

        self.atm = DinoAtmosphere()
        self.lin_to_gauss, self.gauss_to_lin = dc.make_regridders_jax(self.atm.lat_deg)

        # ocean grid metrics (as in run_dino_coupled / main)
        lat = np.linspace(-90, 90, OCEAN_GRID.nlat)
        self.dy = (np.pi * EARTH_RADIUS) / OCEAN_GRID.nlat
        cos_lat = np.maximum(np.cos(np.deg2rad(lat)), 0.05)
        self.dx = jnp.asarray((2 * np.pi * EARTH_RADIUS * cos_lat[:, None]) / OCEAN_GRID.nlon)
        self.dz = jnp.asarray(OCEAN_DZ)

        # land forcing (annual-ish downward shortwave; perpetual day=80 as in harness)
        self.land_mask_f = (~np.asarray(self.omask)).astype(np.float32)
        self.ocean_mask_f = jnp.asarray(np.asarray(self.omask).astype(np.float32))
        insol = np.maximum(np.asarray(aphys.compute_solar_insolation(
            jnp.asarray(lat) * np.pi / 180.0, day_of_year=80.0)), 0.0)[:, None]
        self.sw_down = jnp.asarray(np.broadcast_to(insol, (OCEAN_GRID.nlat, OCEAN_GRID.nlon)))

        # remat'd atmosphere interval (n_steps static) so the adjoint fits in memory
        # (used ONLY by the differentiable step(); the forward step_fast does not remat).
        self._atmos_remat = jax.checkpoint(self.atm._run_interval, static_argnums=(2,))

        # FAST forward step: the WHOLE coupling interval jitted as one fused program
        # (no remat), at a fixed interval. This is ~10x faster than the eager step()
        # for control/forced runs -- eager dispatch of the spectral transforms, regrid
        # and the 96-step ocean scan per interval was the slowdown. Differentiable
        # step() (remat, variable interval) is kept for gradients / DA.
        self.interval = interval
        self._n_atm_f = int(round(self.atm.steps_per_day * interval))
        self._n_sub_f = int(round(86400.0 * interval / DT_OCEAN))
        self._step_fast = jax.jit(
            lambda cs, co2, hose: self._advance(cs, co2, self.interval, self._n_atm_f,
                                                self._n_sub_f, remat=False, hosing_sv=hose))

    # ---- state ----
    def init_state(self):
        sst0_g = self.lin_to_gauss(jnp.asarray(self._ocean0.temp[0]))
        ice0 = init_ice_state(OCEAN_GRID.nlat, OCEAN_GRID.nlon)
        return DinoCoupledState(ocean=self._ocean0,
                                atmos=self.atm.initial_state(sst0_g),
                                land=self._land0, ice=ice0, day=0.0)

    # ---- one coupling interval (shared body; jit-friendly) ----
    def _advance(self, cstate, co2_ppm, interval, n_atm, n_sub, remat, hosing_sv=0.0):
        atm = self.atm
        sst_lin = cstate.ocean.temp[0]
        # ice-modified ocean surface temp seen by the atmosphere (lagged ice):
        # concentration-weighted blend of open-water SST and the ice skin temp.
        A_prev = cstate.ice.concentration
        sst_ocean_ice = (1.0 - A_prev) * sst_lin + A_prev * (cstate.ice.surface_temp + 273.15)
        # lower boundary: ice-modified SST over sea, land skin T over land (lagged).
        surf_T = jnp.where(self.omask, sst_ocean_ice, cstate.land.temp)
        sst_g = self.lin_to_gauss(surf_T)

        # atmosphere interval (nondim T == K, so SST passes through directly).
        # remat ONLY when differentiating; forward runs skip it (it blocks fusion).
        atmos_fn = self._atmos_remat if remat else atm._run
        dino_state = atmos_fn(cstate.atmos, sst_g, n_atm)

        diag = dc.dino_diagnostics_jax(atm, dino_state)
        u_sfc = self.gauss_to_lin(diag["u_sfc"])
        v_sfc = self.gauss_to_lin(diag["v_sfc"])
        t_air = self.gauss_to_lin(diag["t_sfc"])
        q_air = self.gauss_to_lin(diag["q_sfc"])
        precip_a = self.gauss_to_lin(diag["precip"])

        # land surface
        lw_down_l = 0.8 * 5.67e-8 * jnp.maximum(t_air - 10.0, 150.0) ** 4
        wind_sp = jnp.sqrt(u_sfc ** 2 + v_sfc ** 2) + 1.0
        new_land, _ = step_land(
            cstate.land, t_air=t_air, q_air=q_air, sw_down=self.sw_down,
            lw_down=lw_down_l, precip=jnp.maximum(precip_a, 0.0) * 1e-3,
            mask=jnp.asarray(self.land_mask_f), wind_speed=wind_sp,
            drag_coeff=DRAG_COEFF_LAND, dt=86400.0 * interval)

        # sea ice (Semtner thermodynamic): forced by this interval's atmosphere,
        # returns heat + freshwater fluxes to the ocean. Units: t_air/SST in degC.
        new_ice, (ice_heat, ice_fw) = step_ice(
            cstate.ice, t_air=t_air - 273.15, sw_down=self.sw_down,
            lw_down=lw_down_l, ocean_temp=sst_lin - 273.15,
            ny=OCEAN_GRID.nlat, nx=OCEAN_GRID.nlon, mask=self.ocean_mask_f)

        # bulk surface fluxes (jnp), blended with the ice fluxes by concentration.
        nh, fw, tx, ty = dc.ocean_fluxes_jax(
            sst_lin, u_sfc, v_sfc, t_air, q_air, precip_a,
            ocean_mask=self.omask, sst_target=self.sst_target,
            restore_tau_days=self.restore_tau_days, q_flux=self.q_flux, co2_ppm=co2_ppm)
        A = new_ice.concentration
        nh = (1.0 - A) * nh + A * ice_heat
        fw = (1.0 - A) * fw + A * ice_fw
        fluxes = (nh, fw, jnp.zeros_like(nh))
        wind = (tx, ty)

        def _ocean(oc, _):
            return veros_driver.step_ocean(
                oc, surface_fluxes=fluxes, wind_stress=wind, dx=self.dx, dy=self.dy,
                dz=self.dz, nz=self.nz, mask=self.surface_mask,
                ocean_mask_3d=self.ocean_mask_3d, thc_haline_gain=self.thc_haline_gain,
                thc_contrast_depth_m=self.thc_contrast_depth_m,
                hosing_sv=hosing_sv), None
        body = jax.checkpoint(_ocean) if remat else _ocean
        new_ocean, _ = jax.lax.scan(body, cstate.ocean, None, length=n_sub)
        return DinoCoupledState(ocean=new_ocean, atmos=dino_state, land=new_land,
                                ice=new_ice, day=cstate.day + interval)

    # ---- differentiable, variable-interval step (eager; for gradients / DA) ----
    def step(self, cstate, interval=1.0, co2_ppm=None, hosing_sv=0.0):
        n_atm = int(round(self.atm.steps_per_day * interval))
        n_sub = int(round(86400.0 * interval / DT_OCEAN))
        return self._advance(cstate, co2_ppm, interval, n_atm, n_sub, remat=True,
                             hosing_sv=hosing_sv)

    # ---- fast, fully-jitted forward step (fixed interval; control / forced runs) ----
    def step_fast(self, cstate, co2_ppm=280.0, hosing_sv=0.0):
        """One coupling interval as a single fused jitted program (no remat). ~10x
        faster than step() for forward integration. co2_ppm=280 -> zero forcing;
        hosing_sv freshens the subpolar N. Atlantic (AMOC tipping forcing)."""
        return self._step_fast(cstate, jnp.asarray(float(co2_ppm)),
                               jnp.asarray(float(hosing_sv)))

    # ---- surface diagnostics on the linear grid (for scoring / inspection) ----
    def diagnostics_lin(self, cstate):
        diag = dc.dino_diagnostics_jax(self.atm, cstate.atmos)
        return {k: self.gauss_to_lin(diag[k])
                for k in ("u_sfc", "v_sfc", "t_sfc", "q_sfc", "precip", "mslp")}


# --------------------------------------------------------------------------- #
# Checkpoint / resume (state-complete, bit-exact).
# --------------------------------------------------------------------------- #
def _save_nt(blob, prefix, nt):
    for f in nt._fields:
        blob[f"{prefix}__{f}"] = np.asarray(getattr(nt, f))


def _load_nt(d, prefix, cls):
    return cls(**{f: jnp.asarray(d[f"{prefix}__{f}"]) for f in cls._fields})


def save_state(cstate, path_base):
    """Checkpoint a DinoCoupledState to ``<path_base>.npz`` (ocean/land/ice arrays
    + day) plus ``<path_base>_dino.npz`` (the dinosaur modal state, via the dycore's
    own save). Saves ALL ocean fields (incl. the diagnostic psi/rho/w warm-starts),
    so a reload is bit-exact -- resume is prognostically continuous, not just
    state-complete (unlike the legacy nc harness, which recomputes diagnostics)."""
    blob = {}
    _save_nt(blob, "ocean", cstate.ocean)
    _save_nt(blob, "land", cstate.land)
    _save_nt(blob, "ice", cstate.ice)
    blob["day"] = np.asarray(float(cstate.day))
    np.savez(path_base + ".npz", **blob)
    dino_save(cstate.atmos, path_base + "_dino.npz")


def load_state(path_base):
    """Inverse of :func:`save_state`. The dino modal state restores ``sim_time=None``
    (the dycore convention; a scalar would break the first resumed ImEx step)."""
    d = np.load(path_base + ".npz", allow_pickle=False)
    return DinoCoupledState(
        ocean=_load_nt(d, "ocean", OceanState),
        atmos=dino_load(path_base + "_dino.npz"),
        land=_load_nt(d, "land", LandState),
        ice=_load_nt(d, "ice", IceState),
        day=float(d["day"]))


__all__ = ["DinoCoupledState", "DinoCoupledModel", "save_state", "load_state"]
