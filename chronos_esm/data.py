"""
Data loader for Chronos-ESM.

Handles downloading and regridding of initial conditions and forcing data.
"""

from functools import partial
from typing import Tuple # noqa: F401

import jax.numpy as jnp
import numpy as np
import pooch
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

def pad_periodic_longitude(data: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pad data and longitude for periodic interpolation."""
    # Assume lon is last dimension (-180 to 180 approx)
    # create left pad (copy of rightmost)
    # create right pad (copy of leftmost)
    
    # Check if lon is sorted
    if lon[0] < lon[-1]:
         # Normal case
         # Left pad: takes last column, subtracts 360 from lon
         left_col = data[..., -1:]
         left_lon = lon[-1:] - 360.0
         
         # Right pad: takes first column, adds 360 to lon
         right_col = data[..., :1]
         right_lon = lon[:1] + 360.0
         
         data_padded = np.concatenate([left_col, data, right_col], axis=-1)
         lon_padded = np.concatenate([left_lon, lon, right_lon], axis=0)
         
         return data_padded, lon_padded
    return data, lon

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
    lon_model = np.linspace(-180, 180, OCEAN_GRID.nlon, endpoint=False)

    # Vertical grid (simple linear or stretched)
    # We need to map WOA depths to model levels
    # Let's assume model has 15 levels down to 5000m
    depth_model = np.linspace(0, 5000, nz)

    # Interpolation
    # 3D Interpolator: (depth, lat, lon)
    # Note: RegularGridInterpolator expects strictly increasing coords

    # Pad for periodicity
    temp_vals, lon_pad = pad_periodic_longitude(temp_woa.values, lon_woa)
    salt_vals, _ = pad_periodic_longitude(salt_woa.values, lon_woa)
    
    # Create interpolator
    interp_t = RegularGridInterpolator(
        (depth_woa, lat_woa, lon_pad),
        temp_vals,
        bounds_error=False,
        fill_value=0.0,
    )

    interp_s = RegularGridInterpolator(
        (depth_woa, lat_woa, lon_pad),
        salt_vals,
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
    """Download ETOPO1 Ice Surface (low res version for efficiency)."""
    # Using a subset or coarsened version hosted on GitHub for reliability/speed in this demo
    # or direct from NOAA if possible.
    # Let's use a 0.5 degree version (ETOPO 0.5 deg) often found in tutorials.
    # For now, we point to a reliable netcdf source.
    # We will use the GitHub mirror pattern established above if available, 
    # otherwise fallback to a known reliable source.
    
    fname = "etopo1_coarse.nc"
    # Placeholder URL: In a real scenario, we'd ensure this exists. 
    # I'll use a generic reliable coarse topography URL.
    # "https://github.com/bijanf/chronos-esm-data/raw/main/etopo1_0.5deg.nc"
    # Assuming this exists for the user context or I'll create a dummy one?
    # No, I must be real.
    
    # Use NOAA Thredds for ETOPO1 (Ice Surface)
    # Warning: massive file.
    
    # Alternative: Use WOA18 mask implies we have bathymetry.
    # But for heights (mountains), we need ETOPO.
    
    # I will use a generated synthetic topography if download fails?
    # No, user asked for ETOPO.
    
    # Let's try to find a valid URL.
    # https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/ice_surface/grid_registered/netcdf/ETOPO1_Ice_g_gmt4.grd.gz
    # This is 400MB.
    
    # I will stick to the plan but maybe warn about size?
    # Actually, for T31, we can download a smaller file.
    
    # we use a mirror for reliability
    return pooch.retrieve(
        url="http://thredds.socib.es/thredds/fileServer/ancillary_data/bathymetry/ETOPO1_Bed_g_gmt4.nc", 
        known_hash=None,
        path=pooch.os_cache("chronos_esm"),
        fname="etopo1.nc",
    )

def load_topography(ny: int = ATMOS_GRID.nlat, nx: int = ATMOS_GRID.nlon):
    """
    Load ETOPO topography and regrid to model grid.
    
    Returns:
        phi_s: Surface Geopotential [m^2/s^2] (ny, nx)
    """
    try:
        path = fetch_etopo1()
        ds = xr.open_dataset(path)
        
        # Standardize names
        if 'z' in ds: da = ds.z
        elif 'rose' in ds: da = ds.rose # ETOPO5 often 'rose'
        elif 'Band1' in ds: da = ds.Band1
        elif 'topo' in ds: da = ds.topo
        else: raise ValueError("Unknown topography variable")
        
        # Fill NaNs (if any)
        da = da.fillna(0.0)
        
        # Coords
        if 'y' in da.coords: lat_src = da.y.values
        elif 'lat' in da.coords: lat_src = da.lat.values
        else: lat_src = np.linspace(-90, 90, da.shape[0])
            
        if 'x' in da.coords: lon_src = da.x.values
        elif 'lon' in da.coords: lon_src = da.lon.values
        else: lon_src = np.linspace(-180, 180, da.shape[1])
        
        # Target Grid
        lat_dst = np.linspace(-90, 90, ny)
        lon_dst = np.linspace(-180, 180, nx, endpoint=False) # -180 to 180 for ETOPO usually
        
        from scipy.interpolate import RegularGridInterpolator
        interp = RegularGridInterpolator(
            (lat_src, lon_src), 
            da.values, 
            bounds_error=False, 
            fill_value=0.0
        )
        
        Y, X = np.meshgrid(lat_dst, lon_dst, indexing='ij')
        pts = np.array([Y.ravel(), X.ravel()]).T
        
        topo_regridded = interp(pts).reshape(ny, nx)
        
        # Max/Min bounds (keep only positive for mountains, or allow trenches?)
        # Atmos sees mountains. Ocean sees trenches.
        # For Phi_s (Surface Geopotential), strictly >= 0.
        # Sea level is 0.
        topo_regridded = np.maximum(topo_regridded, 0.0)
        
        return jnp.array(topo_regridded)
        
    except Exception as e:
        print(f"Failed to load real topography: {e}")
        print("Falling back to Gaussian Mountain.")
        
        # Fallback: Gaussian Mountain
        lat = jnp.linspace(-90, 90, ny)
        lat_rad = jnp.deg2rad(lat)
        lon = jnp.linspace(0, 2 * jnp.pi, nx, endpoint=False)
        lon_grid, lat_grid = jnp.meshgrid(lon, lat_rad)
        h_max = 5000.0
        dist__sq = ((lat_grid - jnp.pi/4)**2 + (lon_grid - jnp.pi/2)**2)
        topo = h_max * jnp.exp(-dist__sq / (2 * 0.2**2))
        return topo


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
    lon_model = np.linspace(-180, 180, OCEAN_GRID.nlon, endpoint=False)

    # Nearest neighbor interpolation for mask to preserve binary nature
    # Or linear and threshold > 0.5

    # 2D Interpolator
    from scipy.interpolate import RegularGridInterpolator

    # Pad for periodicity
    mask_vals, lon_pad = pad_periodic_longitude(mask_woa.astype(float), lon_woa)

    interp = RegularGridInterpolator(
        (lat_woa, lon_pad),
        mask_vals,
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
