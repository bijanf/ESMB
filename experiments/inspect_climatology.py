
import xarray as xr
import numpy as np
import glob
import os
import sys
import pooch
from scipy.interpolate import RegularGridInterpolator

# Paste the mask loading function to ensure identical logic
def fetch_woa18_temp():
    fname_temp = "woa18_decav_t00_5d.nc"
    path_temp = pooch.retrieve(
        url="https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/temperature/decav/5deg/woa18_decav_t00_5d.nc",
        known_hash=None,
        path=pooch.os_cache("chronos_esm"),
        fname=fname_temp,
    )
    return path_temp

def load_bathymetry_mask_standalone(nlat=48, nlon=96):
    path_t = fetch_woa18_temp()
    ds_t = xr.open_dataset(path_t, decode_times=False)
    temp_surf = ds_t.t_mn.isel(time=0, depth=0)
    mask_woa = ~np.isnan(temp_surf.values)
    lat_woa = ds_t.lat.values
    lon_woa = ds_t.lon.values
    lat_model = np.linspace(-90, 90, nlat)
    lon_model = np.linspace(-180, 180, nlon)
    interp = RegularGridInterpolator(
        (lat_woa, lon_woa),
        mask_woa.astype(float),
        bounds_error=False,
        fill_value=0.0,
        method="nearest",
    )
    Y, X = np.meshgrid(lat_model, lon_model, indexing="ij")
    pts = np.array([Y.ravel(), X.ravel()]).T
    mask_interp = interp(pts).reshape(nlat, nlon)
    return mask_interp > 0.5

def inspect_sst_quality():
    print("Inspecting SST Climatology Quality...")
    
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
    else:
        pattern = 'outputs/control_run_prod_v1/mean_*.nc'
        
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        print(f"No files found for pattern: {pattern}")
        return

    if len(files) < 60:
        print(f"Warning: Using all {len(files)} files found.")
        climatology_files = files
    else:
        climatology_files = files[-60:]
        
    print(f"Loading {len(climatology_files)} files...")
    
    # Manual Mean Computation
    ds_sum = xr.open_dataset(climatology_files[0], decode_times=False)
    # Load SSS (surface salinity)
    # Variable 'ocean_salt' (time, z, y, x)
    # sst_sum = ds_sum.ocean_temp.isel(z=0).values 
    # Use generic name 'field_sum' to avoid confusion, but focusing on SSS
    field_sum = ds_sum.ocean_salt.isel(z=0).values
    
    for f in climatology_files[1:]:
        d = xr.open_dataset(f, decode_times=False)
        field_sum += d.ocean_salt.isel(z=0).values
        
    field_mean_raw = field_sum / len(climatology_files)
    
    # Apply Mask
    ny, nx = field_mean_raw.shape
    mask = load_bathymetry_mask_standalone(nlat=ny, nlon=nx)
    field_mean = np.where(mask, field_mean_raw, np.nan)
    
    # Stats (NaN-aware)
    print("-" * 30)
    print(f"SSS Mean Stats (Ocean Only):")
    print(f"  Min: {np.nanmin(field_mean):.2f} psu")
    print(f"  Max: {np.nanmax(field_mean):.2f} psu")
    print(f"  Avg: {np.nanmean(field_mean):.2f} psu")
    
    # Noise/Roughness Check (Laplacian)
    # Simple discrete laplacian: 4*center - neighbors
    # Ignoring boundaries for simple stats
    inner = field_mean[1:-1, 1:-1]
    up = field_mean[2:, 1:-1]
    down = field_mean[:-2, 1:-1]
    left = field_mean[1:-1, :-2]
    right = field_mean[1:-1, 2:]
    
    # Only calculate where All 5 points are valid (Ocean points not near coast)
    valid_mask = (~np.isnan(inner)) & (~np.isnan(up)) & (~np.isnan(down)) & (~np.isnan(left)) & (~np.isnan(right))
    
    laplacian = np.abs(4 * inner - up - down - left - right)
    laplacian = np.where(valid_mask, laplacian, np.nan)
    
    print("-" * 30)
    print(f"Roughness (Laplacian Magnitude) Stats (Ocean Interior):")
    print(f"  Max Roughness: {np.nanmax(laplacian):.4f} psu")
    print(f"  Mean Roughness: {np.nanmean(laplacian):.4f} psu")
    print(f"  Pixels > 0.5 psu Roughness: {np.sum(laplacian > 0.5)}")
    
    # Check for Checkerboard (Nyquist)
    # Difference between neighbors
    diff_x = np.abs(field_mean[:, 1:] - field_mean[:, :-1])
    diff_y = np.abs(field_mean[1:, :] - field_mean[:-1, :])
    
    print("-" * 30)
    print(f"Neighbor Gradients (Ocean):")
    print(f"  Max dx: {np.nanmax(diff_x):.4f} psu")
    print(f"  Max dy: {np.nanmax(diff_y):.4f} psu")
    
    # Locate the worst spot
    if np.any(~np.isnan(laplacian)):
        y_max, x_max = np.unravel_index(np.nanargmax(laplacian), laplacian.shape)
        # shift indices back because we sliced inner
        y_bad = y_max + 1
        x_bad = x_max + 1
        
        print("-" * 30)
        print(f"Worst Roughness at Index (y={y_bad}, x={x_bad}): {np.nanmax(laplacian):.4f} psu")
        print(f"Value there: {field_mean[y_bad, x_bad]:.2f} psu")
        print(f"Neighbors: U={field_mean[y_bad+1, x_bad]:.2f}, D={field_mean[y_bad-1, x_bad]:.2f}, L={field_mean[y_bad, x_bad-1]:.2f}, R={field_mean[y_bad, x_bad+1]:.2f}")

if __name__ == "__main__":
    inspect_sst_quality()
