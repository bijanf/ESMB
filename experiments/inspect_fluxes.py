
import sys
import xarray as xr
import numpy as np

def inspect_fluxes(path):
    ds = xr.open_dataset(path, decode_times=False)
    print(f"File: {path}")
    print("Variables:", list(ds.data_vars))
    
    for var in ds.data_vars:
        if "flux" in var or "sst" in var or "precip" in var:
            data = ds[var].values
            print(f"{var}: Min={np.nanmin(data):.2e}, Max={np.nanmax(data):.2e}, Mean={np.nanmean(data):.2e}")

    # Also check Ocean Temp again
    if "ocean_temp" in ds:
         print(f"Ocean Temp: Min={np.nanmin(ds.ocean_temp.values):.2e}, Max={np.nanmax(ds.ocean_temp.values):.2e}")

if __name__ == "__main__":
    inspect_fluxes(sys.argv[1])
