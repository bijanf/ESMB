
import xarray as xr
import numpy as np
import sys

def check_file(path):
    try:
        ds = xr.open_dataset(path, decode_times=False)
        print(f"Checking {path}...")
        
        # Check NaNs
        for var in ['atmos_temp', 'ocean_temp', 'ocean_u', 'ocean_v']:
            val = ds[var].values
            if np.isnan(val).any():
                print(f"!! NaN detected in {var} !!")
                return False
            print(f"{var}: Min={val.min():.2f}, Max={val.max():.2f}")
            
        print("File looks healthy.")
        return True
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return False

if __name__ == "__main__":
    path = sys.argv[1]
    check_file(path)
