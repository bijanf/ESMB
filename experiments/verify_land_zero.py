
import xarray as xr
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from chronos_esm import data

def check_land_zero(filename):
    print(f"Checking {filename}...")
    ds = xr.open_dataset(filename, decode_times=False)
    
    # Load Mask (1=Ocean, 0=Land)
    mask = data.load_bathymetry_mask()
    mask = np.array(mask)
    
    # Get SST (Layer 0)
    # Model Temp is Kelvin. I masked it to 0.0.
    # So Land Value should be 0.0.
    
    temp = ds.ocean_temp.isel(z=0).values
    
    # Check Land Points (mask == 0)
    land_vals = temp[mask == 0]
    
    max_land = np.max(np.abs(land_vals))
    print(f"Max Absolute Temp on Land: {max_land}")
    
    if max_land < 1e-9:
        print("SUCCESS: Land is strictly zero.")
    else:
        print("FAILURE: Land has non-zero values.")
        print(f"Sample Land Values: {land_vals[:10]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    check_land_zero(args.filename)
