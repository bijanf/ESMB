
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import sys

def plot_amoc(run_name):
    # Setup
    output_dir = f"outputs/{run_name}"
    files = sorted(glob.glob(os.path.join(output_dir, "mean_*.nc")))
    
    if not files:
        print(f"No files found in {output_dir}")
        return

    print(f"Found {len(files)} files. Calculating AMOC time series...")
    
    amoc_series = []
    months = []
    
    # Constants
    EARTH_RADIUS = 6.371e6
    
    # Process files
    for i, f in enumerate(files):
        try:
            with xr.open_dataset(f, decode_times=False, engine='netcdf4') as ds:
                if 'ocean_v' not in ds:
                    continue
                
                v = ds['ocean_v'].values
                # Handle possible time dimension if xarray didn't squeeze it
                if v.ndim == 4: v = v[0]
                
                nz, ny, nx = v.shape
                # Grid
                dz = 5000.0 / nz
                lat_vals = np.linspace(-90, 90, ny)
                lat_rad = np.deg2rad(lat_vals)
                
                dx = 2 * np.pi * EARTH_RADIUS * np.cos(lat_rad) / nx
                dx = np.broadcast_to(dx[:, None], (ny, nx))
                
                # Zonal Integral
                transport_zonal = np.sum(v * dx[None, :, :], axis=2) # (nz, ny)
                
                # Vertical Integral (Streamfunction) from top down
                moc = -np.cumsum(transport_zonal * dz, axis=0) # (nz, ny)
                moc_sv = moc / 1.0e6
                
                # Extract 20N - 60N Max
                idx_20n = int(ny * (20+90)/180)
                idx_60n = int(ny * (60+90)/180)
                
                # Subset and Max
                moc_subset = moc_sv[:, idx_20n:idx_60n]
                amoc_max = np.max(moc_subset)
                
                amoc_series.append(amoc_max)
                # Extract month from filename "mean_0012.nc"
                m_str = os.path.basename(f).split('_')[1].split('.')[0]
                months.append(int(m_str))
                
                if int(m_str) % 12 == 0:
                    print(f"Processed Month {m_str}")
                    
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # Plot 1: Time Series
    plt.figure(figsize=(10, 6))
    plt.plot(months, amoc_series, marker='o', linestyle='-', markersize=2, label='Max AMOC (20N-60N)')
    plt.axhline(y=17.0, color='r', linestyle='--', alpha=0.5, label='Target (~17 Sv)')
    
    plt.title(f'AMOC Strength Evolution\nRun: {run_name}')
    plt.xlabel('Simulation Month')
    plt.ylabel('Overturning Strength (Sv)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    ts_file = f'amoc_timeseries_{run_name}.png'
    plt.savefig(ts_file)
    print(f"Saved time series to {ts_file}")

    # Plot 2: Latest MOC Streamfunction (Lat-Depth)
    if files:
        plt.figure(figsize=(10, 6))
        # Use last calculated 'moc_sv'
        lat_mesh, depth_mesh = np.meshgrid(lat_vals, np.linspace(0, 5000, nz))
        
        # Invert depth for plotting
        im = plt.contourf(lat_mesh, depth_mesh, moc_sv, 20, cmap='RdBu_r')
        plt.colorbar(im, label='Streamfunction (Sv)')
        plt.gca().invert_yaxis()
        
        plt.title(f'MOC Streamfunction (Month {months[-1]})\nRun: {run_name}')
        plt.xlabel('Latitude')
        plt.ylabel('Depth (m)')
        
        map_file = f'moc_map_{run_name}.png'
        plt.savefig(map_file)
        print(f"Saved MOC map to {map_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="production_control")
    args = parser.parse_args()
    
    plot_amoc(args.run_name)
