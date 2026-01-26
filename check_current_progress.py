
import os
import glob
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def compute_amoc_from_v(v, lat_idx=None):
    """
    Compute AMOC index from meridional velocity v (nz, ny, nx).
    v is in m/s.
    """
    # Grid Constants (T31)
    EARTH_RADIUS = 6.371e6
    nz, ny, nx = v.shape
    
    # Latitude Array
    lat = np.linspace(-90, 90, ny)
    lat_rad = np.deg2rad(lat)
    
    # dx varies with latitude: 2*pi*R*cos(lat) / nx
    dx = 2 * np.pi * EARTH_RADIUS * np.cos(lat_rad) / nx
    # Broadcast dx to (ny, nx)
    dx_2d = np.tile(dx[:, np.newaxis], (1, nx))
    
    # v_transport [m^2/s] = v * dx
    v_transport = v * dx_2d
    
    # Zonal Sum [m^2/s] -> (nz, ny)
    v_zonal = np.sum(v_transport, axis=2)
    
    # Vertical Integration
    # dz is constant 333.33m (5000m / 15 levels)
    dz = 5000.0 / 15.0
    
    # MOC [m^3/s] = - Cumsum(v_zonal * dz) from top to bottom
    moc = -np.cumsum(v_zonal * dz, axis=0)
    
    # Convert to Sv
    moc_sv = moc / 1.0e6
    
    # Extract Index at 30N (or lat_idx)
    if lat_idx is None:
        # Find index for ~30N
        lat_idx = int((30.0 - (-90.0)) / 180.0 * ny)
        lat_idx = min(lat_idx, ny - 1)
        
    # Max overturning in the column at this latitude
    amoc_profile = moc_sv[:, lat_idx]
    amoc_index = np.max(amoc_profile)
    
    return amoc_index

def main():
    # UPDATED PATH
    output_dir = "outputs/production_control" 
    files = sorted(glob.glob(os.path.join(output_dir, "mean_*.nc")))
    
    print(f"Found {len(files)} monthly mean files in {output_dir}")
    
    times = []
    t_surf_mean = []
    amoc_indices = []
    
    print("Processing files...")
    # Basic progress bar
    total = len(files)
    
    for i, f in enumerate(files):
        try:
            ds = xr.open_dataset(f, decode_times=False)
            
            # Time from filename
            basename = os.path.basename(f)
            month_str = basename.split('_')[1].split('.')[0]
            month = int(month_str)
            year = month / 12.0
            times.append(year)
            
            # GMST
            if 'atmos_temp' in ds:
                temp = ds['atmos_temp'].values
                t_surf_mean.append(np.mean(temp))
            else:
                t_surf_mean.append(np.nan)
            
            # AMOC
            # Check for 'ocean_v' or 'v'
            v_var = None
            if 'ocean_v' in ds:
                v_var = ds['ocean_v'].values
            elif 'v' in ds:
                v_var = ds['v'].values
                
            if v_var is not None and v_var.ndim == 3:
                amoc = compute_amoc_from_v(v_var)
                amoc_indices.append(amoc)
            else:
                amoc_indices.append(np.nan)

            ds.close()
            
            if i % 20 == 0:
                print(f"Processed Month {month} (Year {year:.1f})...")
                
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Temp
    axes[0].plot(times, t_surf_mean, label='Global Mean Atmos Temp (K)', color='tab:red')
    axes[0].set_ylabel('Temperature (K)')
    axes[0].set_title(f'Control Run Stability (Year 0 - {times[-1]:.1f})')
    axes[0].grid(True)
    axes[0].legend()
    
    # AMOC
    axes[1].plot(times, amoc_indices, label='AMOC Index (Sv) @ 30N', color='tab:blue')
    axes[1].set_ylabel('AMOC (Sv)')
    axes[1].set_xlabel('Year')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('current_amoc_status.png')
    print("Saved current_amoc_status.png")

if __name__ == "__main__":
    main()
