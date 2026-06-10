"""
Generate README/GitHub validation figures (PNG) from a saved model state.

Scores the state against WOA18 (ocean) + ERA5 (atmosphere) and writes:
  docs/figures/sst_validation.png  - model vs WOA18 SST + bias map
  docs/figures/taylor.png          - normalized Taylor diagram, all surface fields
  docs/figures/scorecard.txt       - the numeric scorecard

Usage:
    python experiments/make_readme_figures.py [path/to/state.nc]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # raster backend for PNG
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm import io, data  # noqa: E402
from chronos_esm.validation import obs, grid, metrics, plots  # noqa: E402
from chronos_esm.validation import scorecard as sc  # noqa: E402
from experiments.validate_control import canonical_fields  # noqa: E402

plots.plt.switch_backend("Agg")  # override plots.py's pdf backend for PNG output

OUT = "docs/figures"


def main():
    state_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/century_physics/final_state.nc"
    os.makedirs(OUT, exist_ok=True)

    print(f"Scoring {state_path} ...")
    state = io.load_state_from_netcdf(state_path)
    mf = canonical_fields(state)

    ocean_surface = obs.woa18_surface()
    try:
        era5 = obs.era5_climatology_fields()
    except Exception as e:  # noqa: BLE001
        print(f"ERA5 unavailable ({e}); ocean-only.")
        era5 = None
    obs_spec = sc.assemble_obs(ocean_surface=ocean_surface, era5=era5)

    # Numeric scorecard (reuse the library; it writes PDFs we discard, but
    # returns the rows we want and the taylor entries via figures=False path).
    rows, _ = sc.run_scorecard(
        {k: v for k, v in mf.items() if k in obs_spec}, obs_spec, make_figures=False)
    with open(os.path.join(OUT, "scorecard.txt"), "w") as fh:
        fh.write(sc.format_scorecard(rows))
    print(sc.format_scorecard(rows))

    # Ocean land/sea mask, so the model panels show ocean only (the dynamics
    # also evolve "land" cells; they are excluded from every metric).
    nlat, nlon = mf["sst"].shape
    omask = np.asarray(data.load_bathymetry_mask(nz=15)).astype(bool)
    if omask.ndim == 3:
        omask = omask[0]

    def ocean_only(field):
        return np.where(omask, field, np.nan)

    mlat, mlon = grid.model_lat(nlat), grid.model_lon(nlon)
    woa_sst = grid.regrid_to_model(ocean_surface["sst"], ocean_surface["lat"],
                                   ocean_surface["lon"], mlat, mlon)

    # --- Figure 1: SST model vs WOA18 (ocean-only) ---
    plots.bias_map(ocean_only(mf["sst"]), woa_sst, mlat, mlon,
                   "Sea-surface temperature", "degC",
                   os.path.join(OUT, "sst_validation.png"))

    # --- Figure 2: zonal-mean SST, model vs WOA18 ---
    plots.zonal_mean_plot(metrics.zonal_mean(ocean_only(mf["sst"])),
                          metrics.zonal_mean(woa_sst), mlat,
                          "Zonal-mean SST", "degC",
                          os.path.join(OUT, "sst_zonal.png"))

    print(f"\nWrote figures to {OUT}/")


if __name__ == "__main__":
    main()
