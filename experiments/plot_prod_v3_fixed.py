
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def plot_fixed_run():
    # Find last valid file
    files = sorted(glob.glob("outputs/control_run_prod_v3_fixed/mean_*.nc"))
    if not files:
        print("No files found.")
        return

    # Use the last file
    f = files[-1]
    print(f"Opening {f}...")
    ds = xr.open_dataset(f, decode_times=False)

    # Extract Fields (Snapshot)
    # SST
    sst = ds.ocean_temp.isel(z=0).values - 273.15 # K -> C
    
    # SSS
    sss = ds.ocean_salt.isel(z=0).values
    
    # Laplacian of SSS
    sss_roll_xm = np.roll(sss, 1, axis=1)
    sss_roll_xp = np.roll(sss, -1, axis=1)
    sss_roll_ym = np.roll(sss, 1, axis=0)
    sss_roll_yp = np.roll(sss, -1, axis=0)
    laplacian = np.abs(sss_roll_xp + sss_roll_xm + sss_roll_yp + sss_roll_ym - 4*sss)
    
    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    
    # 1. SST
    im1 = axes[0].imshow(sst, origin='lower', cmap='RdBu_r', vmin=-2, vmax=30)
    axes[0].set_title(f'SST (C) - {f}')
    plt.colorbar(im1, ax=axes[0])
    
    # 2. SSS
    im2 = axes[1].imshow(sss, origin='lower', cmap='viridis', vmin=30, vmax=40)
    axes[1].set_title(f'SSS (PSU) - {f}')
    plt.colorbar(im2, ax=axes[1])
    
    # 3. SSS Laplacian
    im3 = axes[2].imshow(laplacian, origin='lower', cmap='inferno', vmin=0, vmax=10) # Lower vmax to see details
    axes[2].set_title(f'SSS Laplacian (Noise) - {f}')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('prod_v3_fixed_snapshot.png')
    print("Saved prod_v3_fixed_snapshot.png")

if __name__ == "__main__":
    plot_fixed_run()
