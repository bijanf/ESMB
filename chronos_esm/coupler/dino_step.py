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
from chronos_esm.atmos.dino_atmos import DinoAtmosphere
from chronos_esm.config import (OCEAN_DZ, OCEAN_GRID, EARTH_RADIUS, DT_OCEAN,
                                DRAG_COEFF_LAND)
from chronos_esm.land.driver import step_land
from chronos_esm.ice.driver import IceState, init_ice_state, step_ice
from chronos_esm.ocean import veros_driver
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

    def __init__(self, ocean_ic="woa", restore_to_woa=True):
        base = main.init_model(ocean_ic=ocean_ic)
        self._ocean0 = base.ocean
        self._land0 = base.land
        # WOA SST target for the flux-correction (captured from the WOA init).
        self.sst_target = jnp.asarray(base.ocean.temp[0]) if restore_to_woa else None

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

        # remat'd atmosphere interval (n_steps static) so the adjoint fits in memory.
        self._atmos_remat = jax.checkpoint(self.atm._run_interval, static_argnums=(2,))

    # ---- state ----
    def init_state(self):
        sst0_g = self.lin_to_gauss(jnp.asarray(self._ocean0.temp[0]))
        ice0 = init_ice_state(OCEAN_GRID.nlat, OCEAN_GRID.nlon)
        return DinoCoupledState(ocean=self._ocean0,
                                atmos=self.atm.initial_state(sst0_g),
                                land=self._land0, ice=ice0, day=0.0)

    # ---- one differentiable coupling interval ----
    def step(self, cstate, interval=1.0):
        atm = self.atm
        n_atm = int(round(atm.steps_per_day * interval))
        n_sub = int(round(86400.0 * interval / DT_OCEAN))

        sst_lin = cstate.ocean.temp[0]
        # ice-modified ocean surface temp seen by the atmosphere (lagged ice):
        # concentration-weighted blend of open-water SST and the ice skin temp.
        A_prev = cstate.ice.concentration
        sst_ocean_ice = (1.0 - A_prev) * sst_lin + A_prev * (cstate.ice.surface_temp + 273.15)
        # lower boundary: ice-modified SST over sea, land skin T over land (lagged).
        surf_T = jnp.where(self.omask, sst_ocean_ice, cstate.land.temp)
        sst_g = self.lin_to_gauss(surf_T)

        # atmosphere interval (nondim T == K, so SST passes through directly), remat'd.
        dino_state = self._atmos_remat(cstate.atmos, sst_g, n_atm)

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
            ocean_mask=self.omask, sst_target=self.sst_target)
        A = new_ice.concentration
        nh = (1.0 - A) * nh + A * ice_heat
        fw = (1.0 - A) * fw + A * ice_fw
        fluxes = (nh, fw, jnp.zeros_like(nh))
        wind = (tx, ty)

        @jax.checkpoint
        def body(oc, _):
            return veros_driver.step_ocean(
                oc, surface_fluxes=fluxes, wind_stress=wind, dx=self.dx, dy=self.dy,
                dz=self.dz, nz=self.nz, mask=self.surface_mask,
                ocean_mask_3d=self.ocean_mask_3d), None

        new_ocean, _ = jax.lax.scan(body, cstate.ocean, None, length=n_sub)
        return DinoCoupledState(ocean=new_ocean, atmos=dino_state, land=new_land,
                                ice=new_ice, day=cstate.day + interval)

    # ---- surface diagnostics on the linear grid (for scoring / inspection) ----
    def diagnostics_lin(self, cstate):
        diag = dc.dino_diagnostics_jax(self.atm, cstate.atmos)
        return {k: self.gauss_to_lin(diag[k])
                for k in ("u_sfc", "v_sfc", "t_sfc", "q_sfc", "precip", "mslp")}


__all__ = ["DinoCoupledState", "DinoCoupledModel"]
