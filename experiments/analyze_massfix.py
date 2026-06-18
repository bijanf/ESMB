"""Score the mass-conservation corrector gate run.

For every state_d*.nc checkpoint in --dir, report:
  * stability: finite, min/max temp & salt (clip proximity), |current|max
  * climate drift: global-mean SST / SSS over ocean
  * MASS CONSERVATION: max over latitude of the NET zonally+vertically integrated
    meridional transport (Sv). The corrector subtracts a latitude-uniform v so this
    should be ~0; pre-corrector it is O(100s) Sv.
  * AMOC: upper/lower overturning cell at 26.5N + basin max |streamfunction| (Sv)

Usage: venv/bin/python experiments/analyze_massfix.py outputs/masfix_on
"""
import glob
import os
import re
import sys

import numpy as np

from chronos_esm import io, main
from chronos_esm.config import EARTH_RADIUS, OCEAN_DZ, OCEAN_GRID
from chronos_esm.ocean import diagnostics


def net_meridional_transport_sv(v, ocean_mask_3d):
    """Net (zonally + vertically integrated) meridional transport per latitude, Sv."""
    nz, ny, nx = v.shape
    dz = np.asarray(OCEAN_DZ).reshape(nz, 1, 1)
    lat = np.linspace(-90, 90, ny)
    dx = 2 * np.pi * EARTH_RADIUS * np.cos(np.deg2rad(lat)) / nx          # (ny,)
    maskC = np.asarray(ocean_mask_3d)
    transport = v * dx[None, :, None] * dz * maskC                       # m^3/s per cell
    net = np.sum(transport, axis=(0, 2)) / 1.0e6                         # (ny,) Sv
    return net


def main_cli():
    d = sys.argv[1] if len(sys.argv) > 1 else "outputs/masfix_on"
    files = sorted(glob.glob(os.path.join(d, "state_d*.nc")),
                   key=lambda p: int(re.search(r"state_d(\d+)", p).group(1)))
    files = [f for f in files if "_NAN" not in f]
    if not files:
        print(f"no checkpoints in {d}")
        return
    om3, sm = main.ocean_masks(nz=len(OCEAN_DZ))
    om3 = np.asarray(om3)
    surf = np.asarray(sm).astype(bool)
    atl = np.asarray(diagnostics.create_atlantic_mask())

    print(f"{'day':>6} {'SSTglob':>8} {'SSSglob':>8} {'Tmin':>6} {'Tmax':>6} "
          f"{'Smin':>6} {'Smax':>6} {'|cur|':>6} {'netTmax':>8} {'AMOC+':>7} "
          f"{'AMOC-':>7} {'finite':>6}")
    for f in files:
        day = int(re.search(r"state_d(\d+)", f).group(1))
        st = io.load_state_from_netcdf(f)
        oc = st.ocean
        T = np.asarray(oc.temp); S = np.asarray(oc.salt); v = np.asarray(oc.v)
        finite = bool(np.isfinite(T).all() and np.isfinite(S).all()
                      and np.isfinite(v).all())
        sst = np.where(surf, T[0], np.nan)
        sss = np.where(surf, S[0], np.nan)
        net = net_meridional_transport_sv(v, om3)
        netmax = float(np.nanmax(np.abs(net)))
        amoc = diagnostics.compute_amoc(oc, atlantic_mask=atl, ocean_mask=surf)
        print(f"{day:6d} {np.nanmean(sst) - 273.15:8.3f} {np.nanmean(sss):8.3f} "
              f"{float(np.nanmin(T)) - 273.15:6.1f} {float(np.nanmax(T)) - 273.15:6.1f} "
              f"{float(np.nanmin(S)):6.2f} {float(np.nanmax(S)):6.2f} "
              f"{float(np.abs(v).max()):6.3f} {netmax:8.3f} "
              f"{float(amoc['upper_cell_26N']):7.2f} {float(amoc['lower_cell_26N']):7.2f} "
              f"{str(finite):>6}")


if __name__ == "__main__":
    main_cli()
