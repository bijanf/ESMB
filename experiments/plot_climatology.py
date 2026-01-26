
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def plot_climatology(run_name, start_month=390, end_month=401):
    output_dir = f"outputs/{run_name}"
    # Generate file list
    files = []
    for m in range(start_month, end_month + 1):
        f = os.path.join(output_dir, f"mean_{m:04d}.nc")
        if os.path.exists(f):
            files.append(f)
    
    if not files:
        print(f"No files found for months {start_month}-{end_month}")
        return

    print(f"Loading files...")
    
    # Manual Averaging Loop to avoid dask/memory issues
    ds_sum = None
    count = 0
    
    for f in files:
        print(f"Processing {f}...")
        ds = xr.open_dataset(f, decode_times=False)
        if ds_sum is None:
            ds_sum = ds
        else:
            ds_sum = ds_sum + ds
        count += 1
        
    ds_mean = ds_sum / count
    print(f"Computed mean of {count} files.")
    
    # Load Mask
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from chronos_esm import data
    mask = data.load_bathymetry_mask() # (ny, nx)
    mask = np.array(mask)
    
    # Coordinates
    if 'y_ocn' in ds_mean.coords:
        lat = np.linspace(-90, 90, ds_mean.y_ocn.size)
        lon = np.linspace(-180, 180, ds_mean.x_ocn.size)
    else:
        lat = np.arange(96)
        lon = np.arange(192)

    LON, LAT = np.meshgrid(lon, lat)

    # Plot Settings
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    
    # 1. SST
    ax = axes[0]
    sst = ds_mean.ocean_temp.isel(z=0).values - 273.15
    # Apply Mask
    sst_masked = np.where(mask > 0.5, sst, np.nan)
    
    stats = f"Min: {np.nanmin(sst_masked):.2f}, Max: {np.nanmax(sst_masked):.2f}, Std: {np.nanstd(sst_masked):.2f}"
    
    # Auto-scale colorbar to show full range
    im = ax.pcolormesh(LON, LAT, sst_masked, cmap='RdBu_r')
    plt.colorbar(im, ax=ax, label='SST (C)', extend='both')
    ax.set_title(f"SST Climatology (Year {start_month//12})\n{stats}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_facecolor('gray') # Land color

    # 2. SSS
    ax = axes[1]
    sss = ds_mean.ocean_salt.isel(z=0).values
    # Masking for SSS too
    sss_masked = np.where(mask > 0.5, sss, np.nan)
    stats_sss = f"Min: {np.nanmin(sss_masked):.2f}, Max: {np.nanmax(sss_masked):.2f}"
    
    im = ax.pcolormesh(LON, LAT, sss_masked, cmap='viridis')
    plt.colorbar(im, ax=ax, label='SSS (PSU)', extend='both')
    ax.set_title(f"SSS Climatology\n{stats_sss}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_facecolor('gray')

    # 3. SAT (Atmosphere)
    ax = axes[2]
    if 'atmos_temp' in ds_mean:
        val = ds_mean.atmos_temp.isel(lev=0).values if 'lev' in ds_mean.dims else ds_mean.atmos_temp.values
        sat = val - 273.15
        
        im = ax.pcolormesh(LON, LAT, sat, cmap='RdBu_r') # Auto-scale
        plt.colorbar(im, ax=ax, label='SAT (C)', extend='both')
        ax.set_title(f"Surface Air Temp (SAT) Climatology")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        ax.text(0.5, 0.5, "No Atmosphere Data", ha='center')

    plt.tight_layout()
    out_file = f"climatology_{run_name}_Y{start_month//12}.png"
    plt.savefig(out_file)
    print(f"Saved {out_file}")
    
    # Plot Zonal Means
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    sst_zonal = np.mean(sst, axis=1)
    sat_zonal = np.mean(sat, axis=1)
    
    ax2.plot(lat, sst_zonal, label='SST')
    ax2.plot(lat, sat_zonal, label='SAT')
    ax2.set_xlabel("Latitude")
    ax2.set_ylabel("Temperature (C)")
    ax2.set_title("Zonal Mean Temperature Profile")
    ax2.grid(True)
    ax2.legend()
    
    out_file_z = f"zonal_mean_{run_name}_Y{start_month//12}.png"
    plt.savefig(out_file_z)
    print(f"Saved {out_file_z}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="control_run_prod_v4")
    parser.add_argument("--start", type=int, default=390)
    parser.add_argument("--end", type=int, default=401)
    args = parser.parse_args()
    
    plot_climatology(args.run_dir, args.start, args.end)
