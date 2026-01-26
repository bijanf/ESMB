
import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

def compute_amoc_index(ds, lat_target=30.0):
    """
    Compute AMOC index from dataset.
    Approximates dx and assumes dz=100m as per model config.
    """
    # Constants
    EARTH_RADIUS = 6.371e6
    dz = 100.0
    
    # Get Data
    v = ds['ocean_v'].values # (z, y, x)
    if v.ndim == 4: # handle (time, z, y, x)
        v = v[0]
        
    nz, ny, nx = v.shape
    
    # Latitudes
    lat = np.linspace(-90, 90, ny)
    lat_rad = np.deg2rad(lat)
    
    # Grid Spacing dx
    # dx = 2 * pi * R * cos(lat) / nx
    dx = 2 * np.pi * EARTH_RADIUS * np.cos(lat_rad) / nx
    dx = np.broadcast_to(dx[:, np.newaxis], (ny, nx))
    
    # Zonal Transport V * dx [m^2/s]
    # v is [m/s]
    v_transport = v * dx[np.newaxis, :, :] 
    
    # Zonal Sum [m^2/s] -> [m^2/s] (sum over x)
    v_zonal = np.sum(v_transport, axis=2) # (nz, ny)
    
    # Vertical Integration (Top Down) to get Streamfunction
    # Psi(z) = - Integral_z^0 v dz
    # cumsum from top
    moc = -np.cumsum(v_zonal * dz, axis=0)
    
    # Convert to Sv
    moc_sv = moc / 1.0e6
    
    # Find Index for Target Latitude
    lat_idx = np.argmin(np.abs(lat - lat_target))
    
    # Max overturning at this latitude (usually in top 1km or so)
    # Just take max over depth
    amoc_profile = moc_sv[:, lat_idx]
    amoc_index = np.max(amoc_profile)
    
    return amoc_index

def compute_ohc(ds):
    """
    Compute Ocean Heat Content (J).
    OHC = Integral(rho0 * cp * T) dV
    """
    rho0 = 1025.0 # kg/m3
    cp = 3985.0 # J/kg/K
    
    # Grid info
    t = ds['ocean_temp'].values # (z, y, x) in Kelvin
    if t.ndim == 4:
        t = t[0]

    # Convert K to Celsius for anomaly? 
    # Usually OHC is relative to 0K or freezing? 
    # Convention: standard OHC is usually relative to a baseline, but total heat content is fine.
    # We will use Kelvin for Total Heat Content.
    
    nz, ny, nx = t.shape
    
    # Cell volumes
    EARTH_RADIUS = 6.371e6
    lat = np.linspace(-90, 90, ny)
    lat_rad = np.deg2rad(lat)
    
    dx = 2 * np.pi * EARTH_RADIUS * np.cos(lat_rad) / nx
    dx = np.broadcast_to(dx[np.newaxis, :, np.newaxis], (nz, ny, nx)) # (nz, ny, nx)
    
    dy = np.pi * EARTH_RADIUS / ny # Constant dy
    # dz is 100m
    dz = 100.0
    
    dv = dx * dy * dz
    
    # Integrate
    ohc = np.sum(rho0 * cp * t * dv)
    
    return ohc

def compute_global_temp(ds):
    """Compute global mean atmospheric temperature."""
    t = ds['atmos_temp'].values # (y, x)
    if t.ndim == 3:
        t = t[0]
        
    ny, nx = t.shape
    lat = np.linspace(-90, 90, ny)
    weights = np.cos(np.deg2rad(lat))
    weights = np.broadcast_to(weights[:, np.newaxis], (ny, nx))
    
    t_mean = np.sum(t * weights) / np.sum(weights)
    return t_mean

def main():
    data_dir = "outputs/production_control"
    output_plot = os.path.join(data_dir, "steady_state.png")
    
    print(f"Scanning {data_dir}...")
    files = sorted(glob.glob(os.path.join(data_dir, "mean_*.nc")))
    
    if not files:
        print("No files found.")
        return

    times = []
    temps = []
    amocs = []
    ohcs = []
    
    print(f"Found {len(files)} files. Processing...")
    
    for f in files:
        try:
            # Use decode_times=False to avoid calendar issues
            with xr.open_dataset(f, decode_times=False) as ds:
                # Extract Month Index from filename or file
                # filename: mean_XXXX.nc
                basename = os.path.basename(f)
                idx = int(basename.split('_')[1].split('.')[0])
                
                t_mean = compute_global_temp(ds)
                amoc = compute_amoc_index(ds)
                ohc = compute_ohc(ds)
                
                times.append(idx)
                temps.append(t_mean)
                amocs.append(amoc)
                ohcs.append(ohc)
                
                if idx % 10 == 0:
                    print(f"Processed Month {idx}: T={t_mean:.2f}, AMOC={amoc:.2f}, OHC={ohc:.2e}")
                    
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Temperature
    ax1.plot(times, temps, 'b-', linewidth=2)
    ax1.set_ylabel('Global Mean Temp (K)')
    ax1.set_title('Global Mean Surface Temperature')
    ax1.grid(True)
    
    # AMOC
    ax2.plot(times, amocs, 'r-', linewidth=2)
    ax2.set_ylabel('AMOC Index (Sv)')
    ax2.set_title('Atlantic Meridional Overturning Circulation')
    ax2.grid(True)
    
    # OHC
    ax3.plot(times, ohcs, 'g-', linewidth=2)
    ax3.set_ylabel('Ocean Heat Content (J)')
    ax3.set_title('Total Ocean Heat Content')
    ax3.set_xlabel('Time (Months)')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"Plot saved to {output_plot}")

if __name__ == "__main__":
    main()
