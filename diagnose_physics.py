
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

OUTPUT_DIR = "outputs/production_control"

def get_latest_mean_file():
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "mean_*.nc")))
    if not files:
        raise FileNotFoundError(f"No mean files found in {OUTPUT_DIR}")
    return files[-1]

def plot_physics(file_path):
    print(f"Loading {file_path}...")
    ds = xr.open_dataset(file_path, decode_times=False)
    
    # Extract Data
    # Fluxes SST (deg C)
    if 'sst' in ds:
        sst = ds['sst'].values
    elif 'fluxes_sst' in ds:
        sst = ds['fluxes_sst'].values
    else:
        print("SST not found, trying ocean_temp top layer")
        sst = ds['ocean_temp'][0].values - 273.15 # Approx
        
    # Precip
    if 'precip' in ds:
        precip = ds['precip'].values
    elif 'fluxes_precip' in ds:
        precip = ds['fluxes_precip'].values
    else:
        print("Precip variable not found!")
        precip = np.zeros_like(sst)

    # Precip Unit Conversion
    # kg/m2/s -> mm/day
    # 1 kg/m2 = 1 mm
    # * 86400 s/day
    precip_mm_day = precip * 86400.0
    
    # Latitudes
    if 'y_atm' in ds:
        ny = ds.sizes['y_atm']
        lat = np.linspace(-90, 90, ny)
    else:
        lat = np.linspace(-90, 90, sst.shape[0])
        
    lon = np.linspace(0, 360, sst.shape[1])
    
    # --- Plotting ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. SST Map
    ax = axes[0, 0]
    im = ax.imshow(sst, origin='lower', extent=[0, 360, -90, 90], cmap='RdBu_r', vmin=-2, vmax=30)
    ax.set_title("SST (deg C)")
    plt.colorbar(im, ax=ax, orientation='horizontal')
    ax.set_aspect('auto')

    # 2. Precip Map
    ax = axes[0, 1]
    # Use Blues, maybe log scale or cap at 10-15 mm/day
    im = ax.imshow(precip_mm_day, origin='lower', extent=[0, 360, -90, 90], cmap='Blues', vmin=0, vmax=15)
    ax.set_title("Precipitation (mm/day)")
    plt.colorbar(im, ax=ax, orientation='horizontal')
    ax.set_aspect('auto')
    
    # 3. Zonal Mean Precip
    ax = axes[1, 0]
    precip_zonal = np.mean(precip_mm_day, axis=1)
    ax.plot(lat, precip_zonal, label='Zonal Mean Precip')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Precip (mm/day)')
    ax.set_title("Zonal Mean Precipitation")
    ax.set_xlim(-90, 90)
    ax.grid(True)
    # Highlight Equator
    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
    
    # 4. Zonal Mean SST
    ax = axes[1, 1]
    sst_zonal = np.mean(sst, axis=1)
    ax.plot(lat, sst_zonal, color='r', label='Zonal Mean SST')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('SST (deg C)')
    ax.set_title("Zonal Mean SST")
    ax.set_xlim(-90, 90)
    ax.grid(True)
    
    plt.tight_layout()
    out_file = "diagnostics_physics_latest.png"
    plt.savefig(out_file)
    print(f"Saved diagnostics to {out_file}")
    
    # Specific Checks
    max_precip = np.max(precip_mm_day)
    print(f"Max Precip: {max_precip:.2f} mm/day")
    
    equator_idx = int(len(lat) / 2)
    # Check if peak is near equator (ITCZ)
    peak_idx = np.argmax(precip_zonal)
    peak_lat = lat[peak_idx]
    print(f"Peak Zonal Precip at Latitude: {peak_lat:.1f}")

    # Check Humidity
    if 'atmos_q' in ds:
        q = ds['atmos_q'].values
        print(f"Max Specific Humidity: {np.max(q):.2e} kg/kg")
        print(f"Mean Specific Humidity: {np.mean(q):.2e} kg/kg")
    else:
        print("atmos_q not in dataset")

    ds.close()

if __name__ == "__main__":
    latest_file = get_latest_mean_file()
    plot_physics(latest_file)
