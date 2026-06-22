"""
Observation / reanalysis loaders for validation.

  - WOA18 (World Ocean Atlas 2018, 5deg annual climatology): ocean T/S.
    Reuses chronos_esm.data.fetch_woa18_temp (pooch-cached download).
  - ERA5 (Copernicus CDS): near-surface atmosphere climatology. Requires the
    `cdsapi` package and a ~/.cdsapirc with a valid CDS key. Downloads monthly
    means once and caches the NetCDF; subsequent calls reuse the cache.
  - Scalar benchmarks (e.g. RAPID AMOC at 26.5N).

Every loader returns plain NumPy on the dataset's NATIVE grid plus its lat/lon;
regridding onto the model grid is done separately in grid.regrid_to_model.
"""

import os
import zipfile

import xarray as xr

from chronos_esm import data

# ---------------------------------------------------------------------------
# Scalar benchmarks
# ---------------------------------------------------------------------------
# RAPID-MOCHA array, AMOC at 26.5N. Long-term mean ~17 Sv (Smeed et al. 2018).
AMOC_RAPID = {"value": 17.0, "sd": 2.0, "lat": 26.5, "source": "RAPID 26.5N"}


# ---------------------------------------------------------------------------
# Ocean: WOA18
# ---------------------------------------------------------------------------
def woa18_surface():
    """WOA18 surface temperature [degC] and salinity [psu] on native grid."""
    path_t, path_s = data.fetch_woa18_temp()
    ds_t = xr.open_dataset(path_t, decode_times=False)
    ds_s = xr.open_dataset(path_s, decode_times=False)
    sst = ds_t.t_mn.isel(time=0, depth=0).values.astype(float)
    sss = ds_s.s_mn.isel(time=0, depth=0).values.astype(float)
    return {
        "sst": sst,  # degC
        "sss": sss,  # psu
        "lat": ds_t.lat.values.astype(float),
        "lon": ds_t.lon.values.astype(float),
    }


def woa18_3d():
    """WOA18 full-depth temperature [degC] / salinity [psu] on native grid."""
    path_t, path_s = data.fetch_woa18_temp()
    ds_t = xr.open_dataset(path_t, decode_times=False)
    ds_s = xr.open_dataset(path_s, decode_times=False)
    return {
        "temp": ds_t.t_mn.isel(time=0).values.astype(float),  # (depth, lat, lon)
        "salt": ds_s.s_mn.isel(time=0).values.astype(float),
        "depth": ds_t.depth.values.astype(float),
        "lat": ds_t.lat.values.astype(float),
        "lon": ds_t.lon.values.astype(float),
    }


# ---------------------------------------------------------------------------
# Atmosphere: ERA5 (Copernicus CDS)
# ---------------------------------------------------------------------------
ERA5_VARS = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "mean_sea_level_pressure",
]


def _default_cache_dir():
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    d = os.path.join(base, "chronos_esm", "era5")
    os.makedirs(d, exist_ok=True)
    return d


def fetch_era5_climatology(years=range(1991, 2021), grid_deg=2.5, cache_dir=None):
    """Download (once) and cache an ERA5 monthly-means file; return its path.

    Pulls 2m temperature, 10m winds, total precipitation and MSLP at a coarse
    `grid_deg` resolution (default 2.5deg, small + fast) for the requested
    years/all months. The time-averaging to an annual climatology happens in
    era5_climatology_fields(). Requires cdsapi + ~/.cdsapirc.
    """
    import cdsapi

    cache_dir = cache_dir or _default_cache_dir()
    yrs = [int(y) for y in years]
    fname = f"era5_monthly_{yrs[0]}_{yrs[-1]}_{grid_deg}deg.nc"
    target = os.path.join(cache_dir, fname)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": ERA5_VARS,
            "year": [str(y) for y in yrs],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "grid": [grid_deg, grid_deg],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        target,
    )
    return target


def _open_era5_dataset(path):
    """Open an ERA5 download as a single Dataset.

    The new CDS often returns a ZIP containing one NetCDF per step-type stream
    (instantaneous vars + accumulated vars). Detect that, extract the members
    and merge them; otherwise open the file directly.
    """
    if not zipfile.is_zipfile(path):
        return xr.open_dataset(path)

    extract_dir = path + "_extracted"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        members = [m for m in z.namelist() if m.endswith(".nc")]
        z.extractall(extract_dir)
    dsets = [xr.open_dataset(os.path.join(extract_dir, m)) for m in members]
    return xr.merge(dsets, compat="override", join="outer")


def era5_climatology_fields(path=None, **fetch_kwargs):
    """Open a cached ERA5 file and return the annual-mean climatology fields.

    Returns a dict of native-grid arrays in model-comparable units:
      t2m    [K]            2 m air temperature
      u10    [m/s]          10 m zonal wind
      v10    [m/s]          10 m meridional wind
      msl    [Pa]           mean sea-level pressure
      precip [kg/m^2/s]     total precipitation rate
      lat, lon [deg]
    """
    if path is None:
        path = fetch_era5_climatology(**fetch_kwargs)
    ds = _open_era5_dataset(path)

    # Average over all time steps (months x years) -> annual climatology.
    tdim = next(
        (d for d in ("valid_time", "time", "forecast_reference_time") if d in ds.dims),
        None,
    )
    if tdim is not None:
        ds = ds.mean(dim=tdim, keep_attrs=True)

    latname = "latitude" if "latitude" in ds else "lat"
    lonname = "longitude" if "longitude" in ds else "lon"

    def get(*names):
        for n in names:
            if n in ds:
                return ds[n].values.astype(float)
        raise KeyError(
            f"None of {names} found in ERA5 file (have {list(ds.data_vars)})"
        )

    tp = get("tp")  # m (mean daily accumulation in monthly-means)
    precip = tp * 1000.0 / 86400.0  # m/day -> mm/day -> kg/m^2/s

    return {
        "t2m": get("t2m"),  # K
        "u10": get("u10"),  # m/s
        "v10": get("v10"),  # m/s
        "msl": get("msl"),  # Pa
        "precip": precip,  # kg/m^2/s
        "lat": ds[latname].values.astype(float),
        "lon": ds[lonname].values.astype(float),
    }
