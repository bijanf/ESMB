"""Smoke + behaviour tests for the SST-coupled dinosaur multi-level atmosphere.

Requires dinosaur-dycore (see requirements.txt). Skipped if not installed.
"""

import numpy as np
import pytest

pytest.importorskip("dinosaur")

from chronos_esm.atmos.dino_atmos import DinoAtmosphere  # noqa: E402


def test_sst_coupled_spinup():
    """A realistic SST gradient must drive a baroclinic jet and pull the lower
    atmosphere toward SST (warm equator, cold poles), and stay finite."""
    atm = DinoAtmosphere(
        layers=18, orography=False
    )  # clean aquaplanet for the idealized check
    lat = atm.lat_deg
    sst1d = 300.0 - 45.0 * np.sin(np.deg2rad(lat)) ** 2  # 300 K eq -> 255 K pole
    sst = np.broadcast_to(sst1d[None, :], (atm.nlon, atm.nlat))

    state = atm.initial_state(sst)  # near-equilibrium init tailored to this SST
    state = atm.step(state, sst, n_days=15)
    d = atm.diagnostics(state)

    assert np.isfinite(d["u"]).all() and np.isfinite(d["temperature"]).all()

    def band(z, lo, hi):
        m = (lat >= lo) & (lat < hi)
        return float(z[m].mean())

    # lower atmosphere tracks the SST gradient (equator warmer than poles, by a lot)
    tzm = d["t_sfc"].mean(axis=0)
    assert band(tzm, -15, 15) - band(tzm, 60, 90) > 10.0

    # baroclinic upper-level mid-latitude westerly jet develops
    uup = d["u"][atm.layers // 4].mean(axis=0)
    assert band(uup, 30, 60) > 2.0  # NH westerlies
    assert band(uup, -60, -30) > 2.0  # SH westerlies

    # surface tropical easterlies (trades)
    usfc = d["u_sfc"].mean(axis=0)
    assert band(usfc, -20, 20) < 0.0

    # prognostic moisture cycle: humidity builds up and precipitation is in a
    # physically realistic global-mean range (the single-level model was ~1.1
    # mm/day, far too dry; observed ~2.9).
    assert np.isfinite(d["precip"]).all()
    assert float(d["q_sfc"].mean()) > 2e-3  # > 2 g/kg surface humidity
    precip_mmday = d["precip"].mean(axis=0) * 86400.0
    gpr = float(np.average(precip_mmday, weights=np.cos(np.deg2rad(lat))))
    assert 0.5 < gpr < 8.0
