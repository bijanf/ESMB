
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def debug_maps():
    files = sorted(glob.glob('outputs/control_run_prod_v1/mean_*.nc'))
    if not files:
        print("No files found")
        return
    
    # Load last file
    f = files[-1]
    print(f"Inspecting {f}")
    ds = xr.open_dataset(f, decode_times=False)
    
    # Basic Grid Check
    print("Grid Sizes:", ds.sizes)
    if 'y_ocn' in ds:
        print("Lat Ocn first/last:", ds.y_ocn.values[0], ds.y_ocn.values[-1])
    
    # 1. Raw SST
    sst_raw = ds.ocean_temp.isel(z=0).values
    print(f"SST Raw Stats: Min={np.min(sst_raw):.2f}, Max={np.max(sst_raw):.2f}, Mean={np.mean(sst_raw):.2f}")
    print(f"SST Raw NaNs: {np.isnan(sst_raw).sum()}")
    print(f"SST Raw Zeros: {np.sum(sst_raw == 0)}")
    
    plt.figure()
    plt.imshow(sst_raw, origin='lower', cmap='RdYlBu_r')
    plt.colorbar()
    plt.title("Raw SST (No external mask)")
    plt.savefig('debug_sst_raw.png')
    plt.close()
    
    # 2. Generated Mask
    nx = ds.sizes['x_ocn']
    ny = ds.sizes['y_ocn']
    mask = load_bathymetry_mask_standalone(nlat=ny, nlon=nx)
    
    plt.figure()
    plt.imshow(mask, origin='lower', cmap='binary')
    plt.title("Generated Mask (White=True/Ocean)")
    plt.savefig('debug_mask_gen.png')
    plt.close()
    
    # 3. Overlay
    sst_masked = np.ma.masked_where(~mask, sst_raw)
    plt.figure()
    plt.imshow(sst_masked, origin='lower', cmap='RdYlBu_r')
    plt.colorbar()
    plt.title("SST with Generated Mask")
    plt.savefig('debug_sst_masked.png')
    plt.close()
    
    # 4. Compare with Internal Mask?
    # Usually internal mask is where value != 0 or not NaN?
    # If the model initializes land to 0 (273.15? or just 0?)
    # If standard init, land might be 0.
    
    mask_internal = (sst_raw != 0) & (~np.isnan(sst_raw))
    plt.figure()
    plt.imshow(mask_internal, origin='lower', cmap='binary')
    plt.title("Internal Mask (SST != 0 and not NaN)")
    plt.savefig('debug_mask_internal.png')
    plt.close()
    
    # 5. Difference of Masks
    if mask.shape == mask_internal.shape:
        diff_mask = mask.astype(int) - mask_internal.astype(int)
        plt.figure()
        plt.imshow(diff_mask, origin='lower', cmap='seismic')
        plt.colorbar()
        plt.title("Mask Diff (Gen - Internal)")
        plt.savefig('debug_mask_diff.png')
        plt.close()

if __name__ == "__main__":
    debug_maps()
