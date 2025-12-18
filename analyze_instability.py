
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.getcwd())
from chronos_esm import data

def analyze_artifacts():
    # Open latest file for quick check (instability is recent?)
    # Or should we check climatology? Instability might disappear in mean.
    # Instability usually explodes at the end.
    
    import glob
    files = sorted(glob.glob("outputs/production_control/mean_*.nc"))
    last_file = files[-1]
    print(f"Analyzing {last_file}...")
    
    ds = xr.open_dataset(last_file, decode_times=False, engine='netcdf4')
    
    lat = ds['y_atm'].values if 'y_atm' in ds else np.arange(48)
    if 'lat' in ds: lat = ds.lat.values
    
    # 1. South Pole Check
    print("\n--- South Pole Instability ---")
    # Index 3 was reported hot.
    at = ds['atmos_temp'].values[0] if ds['atmos_temp'].ndim==3 else ds['atmos_temp'].values
    sp_temp = at[3, :]
    print(f"Lat Index 3 Mean Temp: {sp_temp.mean():.2f} K")
    print(f"Lat Index 3 Max Temp: {sp_temp.max():.2f} K")
    
    # 2. Equator Check
    # Equator is around index 24 (of 48).
    eq_idx = 24
    print("\n--- Equator Artifacts ---")
    eq_temp_slice = at[eq_idx-2:eq_idx+2, :]
    print(f"Equatorial Mean Temp: {eq_temp_slice.mean():.2f} K")
    
    # Check for "Red Boxes" (Ocean > Atmos)
    mask = np.array(data.load_bathymetry_mask())
    
    ot = ds['ocean_temp'].values
    if ot.ndim==3: ot = ot[0]
    
    # Check diff at equator
    diff = ot - at
    diff_eq = diff[eq_idx-2:eq_idx+2, :]
    mask_eq = mask[eq_idx-2:eq_idx+2, :]
    
    # Filter for Ocean
    diff_ocean_eq = diff_eq[mask_eq]
    
    print(f"Equator Ocean Diff Mean: {diff_ocean_eq.mean():.2f} K")
    print(f"Equator Ocean Diff Max: {diff_ocean_eq.max():.2f} K")
    print(f"Equator Ocean Diff Min: {diff_ocean_eq.min():.2f} K")
    
    # Look for "Red Box" candidates (Diff > 10 K)
    high_diff_indices = np.where((diff_eq > 10) & mask_eq)
    if len(high_diff_indices[0]) > 0:
        print(f"Found {len(high_diff_indices[0])} high-diff (>10K) pixels near Equator.")
        # Plot if needed
    else:
        print("No high-diff artifacts (>10K) found near Equator in snapshot.")

if __name__ == "__main__":
    analyze_artifacts()
