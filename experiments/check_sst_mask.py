
import xarray as xr
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())
try:
    from chronos_esm import data
except ImportError:
    print("Could not import data module")

def check_masking():
    path = 'outputs/control_run_final/mean_0016.nc'
    print(f"Loading {path}...")
    ds = xr.open_dataset(path, decode_times=False, engine='netcdf4')
    
    # Load Land Mask from module if possible, else rely on data
    # In chronos_esm, mask=1 is Ocean, mask=0 is Land usually, or boolean
    try:
        mask = np.array(data.load_bathymetry_mask())
        print(f"Loaded Mask shape: {mask.shape}, Mean: {mask.mean():.2f}")
    except:
        print("Failed to load mask from data module.")
        return

    sst = ds.sst.values
    land_temp = ds.land_temp.values
    
    # Check values on land in SST
    sst_on_land = sst[~mask]
    print(f"SST on Land (Mask=False) -> Mean: {sst_on_land.mean():.2f}, Min: {sst_on_land.min():.2f}, Max: {sst_on_land.max():.2f}")
    
    # Check values on ocean in Land Temp
    land_on_ocean = land_temp[mask]
    print(f"Land Temp on Ocean (Mask=True) -> Mean: {land_on_ocean.mean():.2f}, Min: {land_on_ocean.min():.2f}, Max: {land_on_ocean.max():.2f}")
    
    # Correct Means
    sst_ocean_mean = sst[mask].mean()
    land_land_mean = land_temp[~mask].mean()
    
    print("-" * 30)
    print(f"Corrected Mean SST (Ocean Only): {sst_ocean_mean:.2f} K")
    print(f"Corrected Mean Land Temp (Land Only): {land_land_mean:.2f} K")
    print(f"Global SAT: {ds.atmos_temp.mean().values:.2f} K")
    
    # Check Heat Capacity Logic
    # Month 16 ~ April
    # NH Spring, SH Autumn.
    # Expect Land to be warming up in NH, cooling in SH.
    # Generally, Land has lower heat capacity, so it fluctuates more.
    
    # Sample points
    # Equator (Ocean)
    ny, nx = sst.shape
    eq_idx = ny // 2
    print(f"Equatorial SST (approx): {np.mean(sst[eq_idx, :]):.2f} K")
    
    

if __name__ == "__main__":
    check_masking()
