
import xarray as xr
import numpy as np
import sys

try:
    ds = xr.open_dataset('outputs/control_run_verified/mean_0120.nc', engine='h5netcdf', decode_times=False)
except Exception as e:
    print(f"Failed with h5netcdf: {e}")
    sys.exit(1)

print("Variables:", list(ds.data_vars))

if 'atmos_temp' in ds:
    print(f"ATMOS_TEMP mean: {ds['atmos_temp'].mean().item()}")
    
if 'ocean_temp' in ds:
    print(f"OCEAN_TEMP mean: {ds['ocean_temp'].mean().item()}")
    # Check surface
    if ds['ocean_temp'].ndim >= 3:
        print(f"SST (layer 0) mean: {ds['ocean_temp'][0].mean().item()}")

if 'sst' in ds:
    print(f"SST var mean: {ds['sst'].mean().item()}")

if 'salinity' in ds:
    print(f"SALINITY mean: {ds['salinity'].mean().item()}")
    
if 'salt' in ds:
    print(f"SALT mean: {ds['salt'].mean().item()}")

if 'ocean_salt' in ds:
    print(f"OCEAN_SALT mean: {ds['ocean_salt'].mean().item()}")
