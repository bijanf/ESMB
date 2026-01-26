
import xarray as xr
import numpy as np
import sys

def inspect(filename):
    print(f"Inspecting {filename}...")
    try:
        ds = xr.open_dataset(filename, decode_times=False, engine='h5netcdf')
    except Exception as e:
        print(f"Error opening {filename}: {e}")
        return

    vars_to_check = ['atmos_u', 'atmos_v', 'land_temp', 'atmos_temp', 'ocean_temp']
    
    for v in vars_to_check:
        if v in ds:
            da = ds[v]
            print(f"  {v}: range=[{da.min().values:.4f}, {da.max().values:.4f}], mean={da.mean().values:.4f}, std={da.std().values:.4f}")
            # Check if values are all same (std ~ 0) or zero
            if np.allclose(da.values, 0):
                print(f"  --> {v} is ZERO everywhere!")
            elif da.std().values < 1e-6:
                 print(f"  --> {v} is CONSTANT everywhere!")
        else:
            print(f"  {v} NOT FOUND")

if __name__ == "__main__":
    import glob
    import os
    files = sorted(glob.glob("outputs/production_control/mean_*.nc"))
    if files:
        latest_file = files[-1]
        print(f"Checking latest file: {latest_file}")
        inspect(latest_file)
    else:
        print("No files found in outputs/production_control")
