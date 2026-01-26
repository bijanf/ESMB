
import sys
import xarray as xr
import numpy as np

def locate_max(path, var_name):
    ds = xr.open_dataset(path, decode_times=False)
    data = ds[var_name].values
    
    # Handle NaNs
    if np.all(np.isnan(data)):
        print(f"{var_name} is all NaN")
        return

    max_val = np.nanmax(data)
    min_val = np.nanmin(data)
    
    print(f"Variable: {var_name}")
    print(f"Global Max: {max_val}")
    print(f"Global Min: {min_val}")
    
    # Find indices of max
    # data might be 3D (z, y, x) or 2D (y, x)
    if data.ndim == 3:
        idx = np.unravel_index(np.nanargmax(data), data.shape)
        print(f"Max Location (z, y, x): {idx}")
        # Print surrounding values
        z, y, x = idx
        print(f"Value at max: {data[z, y, x]}")
    elif data.ndim == 2:
        idx = np.unravel_index(np.nanargmax(data), data.shape)
        print(f"Max Location (y, x): {idx}")
        y, x = idx
        print(f"Value at max: {data[y, x]}")

if __name__ == "__main__":
    locate_max(sys.argv[1], "ocean_temp")
