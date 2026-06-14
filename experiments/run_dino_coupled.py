"""Coupled run: dinosaur multi-level atmosphere <-> Chronos ocean (Phase 2b).

Sequential (lagged) coupling at a fixed interval:
  ocean SST  --regrid lin->Gauss-->  DinoAtmosphere.step(SST)  -->  surface winds/T
  surface fields --regrid Gauss->lin--> bulk fluxes (wind stress, SW/LW/sensible/
  latent, net heat) --> step_ocean for the interval.

The atmosphere is the SST-coupled dinosaur dycore (chronos_esm/atmos/dino_atmos).
Moisture is a fixed-RH boundary-layer closure for the surface latent heat (the
prognostic moisture cycle / precip-ITCZ is Phase 3), so the ocean heat budget is
complete enough to be stable while the dynamics are fully baroclinic.

    python experiments/run_dino_coupled.py --days 30
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm import main, io  # noqa: E402
from chronos_esm.ocean import veros_driver  # noqa: E402
from chronos_esm.atmos import physics as aphys  # noqa: E402
from chronos_esm.atmos.dino_atmos import DinoAtmosphere  # noqa: E402
from chronos_esm.config import (OCEAN_GRID, OCEAN_DZ, EARTH_RADIUS, DT_OCEAN,  # noqa: E402
                                ALBEDO_OCEAN)

RHO_AIR, CD = 1.2, 1.3e-3
LAT_LIN = np.linspace(-90, 90, OCEAN_GRID.nlat)


def make_regridders(lat_gauss):
    """1-D latitude interpolation between the linear (ocean/atmos) grid and the
    dinosaur Gaussian grid; longitudes coincide (both 96 pts at 0,3.75,...)."""
    def lin_to_gauss(f_lin):                # (nlat_lin, nlon) -> (nlon, nlat_gauss)
        f = np.asarray(f_lin)
        out = np.stack([np.interp(lat_gauss, LAT_LIN, f[:, j]) for j in range(f.shape[1])], axis=0)
        return out                          # (nlon, nlat_gauss)

    def gauss_to_lin(f_g):                  # (nlon, nlat_gauss) -> (nlat_lin, nlon)
        f = np.asarray(f_g)
        out = np.stack([np.interp(LAT_LIN, lat_gauss, f[j, :]) for j in range(f.shape[0])], axis=1)
        return out                          # (nlat_lin, nlon)
    return lin_to_gauss, gauss_to_lin


_WLAT = np.cos(np.deg2rad(LAT_LIN))[:, None]


def ocean_fluxes(sst_K, u_sfc, v_sfc, t_air_K, q_air, precip_atm, balance_heat=True,
                 ocean_mask=None):
    """Bulk surface fluxes on the linear grid, consistent with the atmosphere's
    own near-surface humidity and precipitation. Returns (net_heat W/m2,
    fw kg/m2/s, tau_x Pa, tau_y Pa)."""
    # annual-mean-ish shortwave into the ocean (insolation * transmission * (1-albedo))
    insol = np.asarray(aphys.compute_solar_insolation(jnp.asarray(LAT_LIN) * np.pi / 180.0,
                                                      day_of_year=80.0))[:, None]
    sw_net = np.maximum(insol, 0.0) * (1.0 - ALBEDO_OCEAN)
    lw_down = 0.8 * 5.67e-8 * np.maximum(t_air_K - 10.0, 150.0) ** 4
    lw_up = 0.98 * 5.67e-8 * sst_K ** 4
    # turbulent fluxes using the ATMOSPHERE's actual near-surface humidity q_air
    # (so latent heat / evaporation are consistent with the moisture budget).
    sens, lat = aphys.compute_surface_fluxes(jnp.asarray(t_air_K), jnp.asarray(q_air),
                                             jnp.asarray(u_sfc), jnp.asarray(v_sfc),
                                             jnp.asarray(sst_K))
    sens, lat = np.asarray(sens), np.asarray(lat)
    net_heat = sw_net + lw_down - lw_up - sens - lat
    if balance_heat:
        # Heat-flux adjustment: remove the area-weighted mean so the control run has
        # no net OCEAN heating/cooling (prevents global SST drift while preserving
        # the spatial flux pattern). Balance over OCEAN cells ONLY: the heat is
        # applied only to ocean cells (via step_ocean's mask), so including land
        # cells -- which carry an unphysical net_heat from the land "SST" values --
        # in the mean leaves a residual net ocean cooling and a spurious cold SST
        # drift. Falling back to the all-cell mean only if no mask is supplied.
        w = np.broadcast_to(_WLAT, net_heat.shape)
        if ocean_mask is not None:
            m = np.asarray(ocean_mask).astype(bool)
            ocean_mean = np.sum(net_heat[m] * w[m]) / np.sum(w[m])
        else:
            ocean_mean = np.sum(net_heat * w) / np.sum(w)
        net_heat = net_heat - ocean_mean
    net_heat = np.clip(net_heat, -1500.0, 1500.0)
    evap = lat / 2.5e6
    fw = precip_atm - evap  # real P - E
    wind_mag = np.maximum(np.sqrt(u_sfc ** 2 + v_sfc ** 2), 1.0)
    tau_x = np.clip(RHO_AIR * CD * wind_mag * u_sfc, -0.3, 0.3)
    tau_y = np.clip(RHO_AIR * CD * wind_mag * v_sfc, -0.3, 0.3)
    return net_heat, fw, tau_x, tau_y


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--interval", type=int, default=1, help="coupling interval [days]")
    ap.add_argument("--save", default="outputs/dino_coupled/final_state.nc")
    args = ap.parse_args()

    state = main.init_model(ocean_ic="woa")
    ocean_mask_3d, surface_mask = main.ocean_masks(nz=state.ocean.u.shape[0])
    atm = DinoAtmosphere()
    lin_to_gauss, gauss_to_lin = make_regridders(atm.lat_deg)
    sst0_g = lin_to_gauss(np.asarray(state.ocean.temp[0]))   # WOA SST on the dino grid
    dino_state = atm.initial_state(sst0_g)                    # near-equilibrium init

    # ocean grid metrics (as in main.py)
    dy_ocn = (np.pi * EARTH_RADIUS) / OCEAN_GRID.nlat
    cos_lat_ocn = np.maximum(np.cos(np.deg2rad(LAT_LIN)), 0.05)
    dx_ocn = jnp.asarray((2 * np.pi * EARTH_RADIUS * cos_lat_ocn[:, None]) / OCEAN_GRID.nlon)
    dz_ocn = jnp.asarray(OCEAN_DZ)
    subs = int(round(86400.0 * args.interval / DT_OCEAN))  # ocean substeps per interval

    @jax.jit
    def ocean_interval(ocean, fluxes, wind):
        def body(oc, _):
            return veros_driver.step_ocean(
                oc, surface_fluxes=fluxes, wind_stress=wind, dx=dx_ocn, dy=dy_ocn,
                dz=dz_ocn, nz=ocean.u.shape[0], mask=surface_mask,
                ocean_mask_3d=ocean_mask_3d), None
        oc, _ = jax.lax.scan(body, ocean, None, length=subs)
        return oc

    wlat = np.cos(np.deg2rad(LAT_LIN))[:, None]
    def gmean(f):
        f = np.asarray(f); w = np.broadcast_to(wlat, f.shape)
        m = np.isfinite(f)  # area-weight only over valid (e.g. ocean) cells
        return float(np.sum(f[m] * w[m]) / np.sum(w[m]))

    print(f"Coupled dinosaur<->ocean: {args.days} days, interval {args.interval}d ({subs} ocean substeps)")
    omask = np.asarray(surface_mask).astype(bool)
    for d in range(args.days // args.interval + 1):
        sst_lin = np.asarray(state.ocean.temp[0])                       # K
        if d > 0:
            sst_g = lin_to_gauss(sst_lin)
            dino_state = atm.step(dino_state, sst_g, n_days=args.interval)
            diag = atm.diagnostics(dino_state)
            u_sfc = gauss_to_lin(diag["u_sfc"]); v_sfc = gauss_to_lin(diag["v_sfc"])
            t_air = gauss_to_lin(diag["t_sfc"]); q_air = gauss_to_lin(diag["q_sfc"])
            precip_a = gauss_to_lin(diag["precip"])
            nh, fw, tx, ty = ocean_fluxes(sst_lin, u_sfc, v_sfc, t_air, q_air, precip_a,
                                          ocean_mask=omask)
            fluxes = (jnp.asarray(nh), jnp.asarray(fw), jnp.zeros_like(jnp.asarray(nh)))
            state = state._replace(ocean=ocean_interval(state.ocean, fluxes, (jnp.asarray(tx), jnp.asarray(ty))))
        diag = atm.diagnostics(dino_state)
        uup = diag["u"][atm.layers // 4].mean(axis=0); lat = atm.lat_deg
        sst_oc = np.where(omask, sst_lin, np.nan)
        jet = float(uup[(np.abs(lat) > 30) & (np.abs(lat) < 60)].mean())
        finite = np.isfinite(diag["u"]).all() and np.isfinite(np.asarray(state.ocean.temp)).all()
        print(f"  day {d*args.interval:3d}: SST {gmean(sst_oc)-273.15:5.2f}C  |u|max {np.abs(diag['u']).max():5.1f}  "
              f"midlat-jet {jet:+5.1f}  |curr|max {float(np.abs(np.asarray(state.ocean.u)).max()):.3f}  "
              f"finite {finite}", flush=True)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    io.save_state_to_netcdf(state, args.save)
    print(f"saved ocean state -> {args.save}")


if __name__ == "__main__":
    main_cli()
