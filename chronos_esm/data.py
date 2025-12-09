"""
Data loader for Chronos-ESM.

Handles downloading and regridding of initial conditions and forcing data.
"""

import os  # noqa: F401

import jax.numpy as jnp
import numpy as np
import pooch
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from chronos_esm.config import ATMOS_GRID, OCEAN_GRID

# Data Registry
DATA_REGISTRY = pooch.create(
    path=pooch.os_cache("chronos_esm"),
    base_url="https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/temperature/decav/5deg/",
    registry={
        "mask": "https://github.com/bijanf/chronos-esm-data/raw/main/"
        "woa18_mask_04.nc",  # noqa: E501
        "temp": "https://github.com/bijanf/chronos-esm-data/raw/main/"
        "woa18_temp_04.nc",  # noqa: E501
        "woa18_decav_t00_5d.nc": None,  # We'll skip hash check for simplicity or add later
    },
)


def fetch_woa18_temp():
    """Download WOA18 5-degree annual temperature climatology."""
    # URL is slightly different for different variables
    # Temp: https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/temperature/decav/5deg/woa18_decav_t00_5d.nc
    # Salt: https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/salinity/decav/5deg/woa18_decav_s00_5d.nc

    # We use a custom fetcher because base_url varies
    fname_temp = "woa18_decav_t00_5d.nc"
    path_temp = pooch.retrieve(
        url="https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/temperature/decav/5deg/woa18_decav_t00_5d.nc",
        known_hash=None,
        path=pooch.os_cache("chronos_esm"),
        fname=fname_temp,
    )

    fname_salt = "woa18_decav_s00_5d.nc"
    path_salt = pooch.retrieve(
        url="https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/salinity/decav/5deg/woa18_decav_s00_5d.nc",
        known_hash=None,
        path=pooch.os_cache("chronos_esm"),
        fname=fname_salt,
    )

    return path_temp, path_salt


def load_initial_conditions(nz: int = 15):
    """
    Load WOA18 data and regrid to model grid.

    Returns:
        temp_ic: (nz, ny, nx) [C]
        salt_ic: (nz, ny, nx) [psu]
    """
    path_t, path_s = fetch_woa18_temp()

    ds_t = xr.open_dataset(path_t, decode_times=False)
    ds_s = xr.open_dataset(path_s, decode_times=False)

    # Extract variables (t_mn: Statistical mean)
    # Dimensions: time, depth, lat, lon
    # We take time=0 (annual mean)
    # Note: WOA18 variable names might be t_mn, t_an, etc. Error suggested t_mn.
    temp_woa = ds_t.t_mn.isel(time=0).fillna(0.0)  # Fill land with 0 for interpolation
    salt_woa = ds_s.s_mn.isel(time=0).fillna(35.0)

    # WOA coordinates
    lat_woa = ds_t.lat.values
    lon_woa = ds_t.lon.values
    depth_woa = ds_t.depth.values

    # Model coordinates (T63)
    # We need to construct the T63 grid lat/lon arrays
    # For now, assume regular grid for regridding target
    lat_model = np.linspace(-90, 90, OCEAN_GRID.nlat)
    lon_model = np.linspace(-180, 180, OCEAN_GRID.nlon)

    # Vertical grid (simple linear or stretched)
    # We need to map WOA depths to model levels
    # Let's assume model has 15 levels down to 5000m
    depth_model = np.linspace(0, 5000, nz)

    # Interpolation
    # 3D Interpolator: (depth, lat, lon)
    # Note: RegularGridInterpolator expects strictly increasing coords

    # Create interpolator
    interp_t = RegularGridInterpolator(
        (depth_woa, lat_woa, lon_woa),
        temp_woa.values,
        bounds_error=False,
        fill_value=0.0,
    )

    interp_s = RegularGridInterpolator(
        (depth_woa, lat_woa, lon_woa),
        salt_woa.values,
        bounds_error=False,
        fill_value=35.0,
    )

    # Create target mesh
    # (nz, ny, nx)
    D, Y, X = np.meshgrid(depth_model, lat_model, lon_model, indexing="ij")

    pts = np.array([D.ravel(), Y.ravel(), X.ravel()]).T

    temp_interp = interp_t(pts).reshape(nz, OCEAN_GRID.nlat, OCEAN_GRID.nlon)
    salt_interp = interp_s(pts).reshape(nz, OCEAN_GRID.nlat, OCEAN_GRID.nlon)

    # Convert to JAX arrays
    return jnp.array(temp_interp), jnp.array(salt_interp)


def fetch_etopo1():
    """Download ETOPO1 bedrock (low res version for testing)."""
    # Using a 0.5 degree or 1 degree version to save bandwidth/time
    # NOAA ETOPO1 Ice Surface
    # URL: https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/grid_registered/netcdf/ETOPO1_Bed_g_gmt4.grd.gz
    # That's huge. Let's use a coarser dataset: ETOPO5 or similar.
    # Or just use the WOA18 mask? WOA18 has land as NaN or fill value.
    # We already have WOA18 loaded. We can derive mask from that!
    # Much easier and consistent with T/S.
    pass


def load_bathymetry_mask(nz: int = 15):
    """
    Derive land mask from WOA18 temperature data.

    Returns:
        mask: (ny, nx) boolean array. True = Ocean, False = Land.
    """
    # Re-use fetch logic (cached)
    path_t, _ = fetch_woa18_temp()
    ds_t = xr.open_dataset(path_t, decode_times=False)

    # Surface temperature
    # In WOA, land is often masked (NaN) or fill value.
    # We used fillna(0.0) in load_initial_conditions.
    # Let's check the raw data for mask.

    # Load raw surface temp
    temp_surf = ds_t.t_mn.isel(time=0, depth=0)

    # Create mask: Valid data is Ocean
    mask_woa = ~np.isnan(temp_surf.values)

    # Regrid mask to model grid
    lat_woa = ds_t.lat.values
    lon_woa = ds_t.lon.values

    lat_model = np.linspace(-90, 90, OCEAN_GRID.nlat)
    lon_model = np.linspace(-180, 180, OCEAN_GRID.nlon)

    # Nearest neighbor interpolation for mask to preserve binary nature
    # Or linear and threshold > 0.5

    # 2D Interpolator
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (lat_woa, lon_woa),
        mask_woa.astype(float),
        bounds_error=False,
        fill_value=0.0,
        method="nearest",
    )

    Y, X = np.meshgrid(lat_model, lon_model, indexing="ij")
    pts = np.array([Y.ravel(), X.ravel()]).T

    mask_interp = interp(pts).reshape(OCEAN_GRID.nlat, OCEAN_GRID.nlon)

    # Threshold
    mask_boolean = mask_interp > 0.5

    return jnp.array(mask_boolean)


if __name__ == "__main__":
    # Test download
    print("Fetching WOA18...")
    t, s = load_initial_conditions()
    print(f"Loaded ICs. Temp shape: {t.shape}, Mean: {jnp.mean(t):.2f}")

    print("Deriving Mask...")
    mask = load_bathymetry_mask()
    print(f"Mask shape: {mask.shape}, Ocean Fraction: {jnp.mean(mask):.2f}")
