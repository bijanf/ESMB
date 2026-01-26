
import jax
import xarray as xr
import numpy as np

try:
    ds = xr.open_dataset("outputs/control_run/mean_0001.nc", decode_times=False)
    print("Loaded mean_0001.nc")
    
    for var in ds.data_vars:
        data = ds[var].values
        if np.any(np.isnan(data)):
            print(f"NaNs found in {var}!")
        print(f"{var}: min={np.nanmin(data):.2e}, max={np.nanmax(data):.2e}, mean={np.nanmean(data):.2e}")
        
except Exception as e:
    print(f"Error: {e}")
