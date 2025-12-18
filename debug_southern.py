
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add path for chronos_esm
sys.path.append(os.getcwd())
from chronos_esm import data

def debug_southern_ocean():
    # Load latest mean file
    import glob
    files = sorted(glob.glob("outputs/production_control/mean_*.nc"))
    if not files:
        print("No files found.")
        return
    latest_file = files[-1]
    print(f"Inspecting {latest_file}...")
    
    ds = xr.open_dataset(latest_file, decode_times=False, engine='netcdf4')
    
    # Load Mask
    mask = np.array(data.load_bathymetry_mask())
    
    # Get Data
    t_surf = None
    if 'sst' in ds: t_surf = ds['sst'].values
    elif 'ocean_temp' in ds: 
        ot = ds['ocean_temp'].values
        t_surf = ot[0] if ot.ndim == 3 else ot
        
    t_atmos = None
    if 'atmos_temp' in ds: t_atmos = ds['atmos_temp'].values
    if t_atmos.ndim == 3: t_atmos = t_atmos[0]
    
    # Subplot focus: Southern Ocean (Lat index 0 to 15 approx, assuming 48 lats)
    # Lat runs -90 to 90. 48 points. approx 3.75 deg per point.
    # South pole is index 0. -60 deg is index ~8.
    
    # Slices
    sl = slice(0, 15)
    
    mask_s = mask[sl, :]
    ts_s = t_surf[sl, :]
    ta_s = t_atmos[sl, :]
    diff_s = ts_s - ta_s
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Mask
    im1 = axes[0,0].imshow(mask_s, origin='lower', cmap='binary')
    axes[0,0].set_title('Land Mask (White=True/Ocean)')
    plt.colorbar(im1, ax=axes[0,0])
    
    # 2. Ocean Temp (Raw)
    im2 = axes[0,1].imshow(ts_s, origin='lower', cmap='viridis')
    axes[0,1].set_title('Raw Ocean Temp')
    plt.colorbar(im2, ax=axes[0,1])
    
    # 3. Atmos Temp
    im3 = axes[1,0].imshow(ta_s, origin='lower', cmap='magma')
    axes[1,0].set_title('Atmos Temp')
    plt.colorbar(im3, ax=axes[1,0])
    
    # 4. Diff (masked)
    diff_masked = np.ma.masked_where(~mask_s, diff_s)
    im4 = axes[1,1].imshow(diff_masked, origin='lower', cmap='RdBu_r', vmin=-20, vmax=20)
    axes[1,1].set_title('Diff (Masked)')
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.savefig('debug_southern_ocean.png')
    print("Saved debug_southern_ocean.png")
    
    # Statistics of the "Red Boxes"
    # Where difference is high positive (>10) AND mask is True (supposedly Ocean)
    # This checks if we have warm ocean pixels that should be cold/ice
    problem_pixels = (diff_s > 10) & mask_s & (ta_s < 260) 
    if np.any(problem_pixels):
        print("\nFound High Diff Pixels in Cold Air regions (Potential Artifacts):")
        indices = np.where(problem_pixels)
        for i, j in zip(indices[0], indices[1]):
            print(f"  Pos({i},{j}): O={ts_s[i,j]:.1f}, A={ta_s[i,j]:.1f}, Diff={diff_s[i,j]:.1f}")
    else:
        print("\nNo obvious high-diff artifacts found with current mask.")

if __name__ == "__main__":
    debug_southern_ocean()
