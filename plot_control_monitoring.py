
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
    # v_zonal index 0 is top (usually). 
    # Let's integrate -cumsum from top.
    moc = -np.cumsum(v_zonal * dz, axis=0)
    
    # Convert to Sv
    moc_sv = moc / 1.0e6
    
    # Extract Index at 30N (or lat_idx)
    if lat_idx is None:
        # Find index for ~30N
        lat_idx = int((30.0 - (-90.0)) / 180.0 * ny)
        lat_idx = min(lat_idx, ny - 1)
        
    # Max overturning in the column at this latitude
    # Usually max value in depth
    amoc_profile = moc_sv[:, lat_idx]
    amoc_index = np.max(amoc_profile)
    
    return amoc_index

def main():
    output_dir = "outputs/control_run"
    files = sorted(glob.glob(os.path.join(output_dir, "mean_*.nc")))
    
    import datetime
    cutoff_time = datetime.datetime(2025, 12, 19, 14, 0).timestamp()
    files = [f for f in files if os.path.getmtime(f) > cutoff_time]
    files = sorted(files)

    print(f"Found {len(files)} monthly mean files.")
    
    times = []
    t_surf_mean = []
    amoc_indices = []
    
    # Check variable names in first file
    try:
        ds0 = xr.open_dataset(files[0], decode_times=False)
        print("Variables in file:", list(ds0.data_vars))
        ds0.close()
    except Exception as e:
        print(f"Error checking first file: {e}")
        return

    print("Processing files...")
    for i, f in enumerate(files):
        try:
            ds = xr.open_dataset(f, decode_times=False)
            
            # Time from filename
            # mean_0001.nc -> Month 1
            basename = os.path.basename(f)
            month_str = basename.split('_')[1].split('.')[0]
            month = int(month_str)
            year = month / 12.0
            times.append(year)
            
            # GMST
            # Variable is 'temp' (Atmos) 
            # Check dimensions. Usually (ny, nx) or (nz, ny, nx)
            if 'atmos_temp' in ds:
                temp = ds['atmos_temp'].values
                if temp.ndim == 3:
                     # Assume last index is surface? Or first?
                     # Let's take mean over all spatial dims for Global Mean Atmos Temp
                     gmst = np.mean(temp) # Volume mean? Or Surface?
                     # Ideally surface. 
                     # If (nz, ny, nx), usually index 0 is top or bottom.
                     # In Chronos code (atmos/dynamics.py), usually 2 levels.
                     # control_run.py -> mean_atmos has 'temp'.
                     pass
                else:
                     gmst = np.mean(temp) 
                
                # If we want exact GMST (Surface Air Temp), we might look for 'fluxes_sst' (Ocean+Ice+Land Temp composite)
                # 'sst' in fluxes is what atmosphere sees.
                # Let's use 'atmos_temp' as a proxy for stability.
                t_surf_mean.append(np.mean(temp))
            
            # AMOC
            # Try to compute from 'ocean_v' (or 'v')
            v_var = None
            if 'ocean_v' in ds:
                v_var = ds['ocean_v'].values
            elif 'v' in ds: # Sometimes saved as v
                v_var = ds['v'].values
            
            if v_var is not None:
                # Expect (nz, ny, nx)
                if v_var.ndim == 3:
                    amoc = compute_amoc_from_v(v_var)
                    amoc_indices.append(amoc)
                else:
                    amoc_indices.append(np.nan)
            else:
                amoc_indices.append(np.nan)

            ds.close()
            
            if i % 12 == 0:
                print(f"Processed Year {year:.1f}")
                
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Temp
    axes[0].plot(times, t_surf_mean, label='Global Mean Atmos Temp (K)', color='tab:red')
    axes[0].set_ylabel('Temperature (K)')
    axes[0].set_title('Control Run Stability (Year 0 - {:.1f})'.format(times[-1]))
    axes[0].grid(True)
    axes[0].legend()
    
    # AMOC
    axes[1].plot(times, amoc_indices, label='AMOC Index (Sv) @ 30N', color='tab:blue')
    axes[1].set_ylabel('AMOC (Sv)')
    axes[1].set_xlabel('Year')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('control_run_diagnostics.png')
    print("Saved control_run_diagnostics.png")

if __name__ == "__main__":
    main()
