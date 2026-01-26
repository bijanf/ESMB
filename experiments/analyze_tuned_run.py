
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys

# Add project root to path for util imports if needed
sys.path.append(os.getcwd())
from chronos_esm import data

def analyze_tuned_run():
    # 1. Setup paths and output files
    input_pattern = "outputs/control_run_tuned_v2/mean_*.nc"
    files = sorted(glob.glob(input_pattern))
    
    if not files:
        print(f"No files found matching {input_pattern}")
        return

    print(f"Found {len(files)} output files. Starting analysis...")

    # Lists to store timeseries data
    t_mean_list = []
    amoc_list = []
    
    # Constants for AMOC
    EARTH_RADIUS = 6.371e6

    def compute_amoc_point(ds):
        """Computes AMOC index at ~30N from a dataset."""
        if 'ocean_v' not in ds: 
            return 0.0
        
        v = ds['ocean_v'].values
        # Handle time dimension if present (time, z, y, x) -> (z, y, x)
        if v.ndim == 4: 
            v = v[0] 
        
        # Grid dimensions from data
        nz, ny, nx = v.shape
        
        # Use simple approximations if grid vars missing
        dz = 5000.0 / nz
        
        # Lat grid approximation: -90 to 90
        lat_vals = np.linspace(-90, 90, ny)
        lat_rad = np.deg2rad(lat_vals)
        
        # dx at each lat
        dx = 2 * np.pi * EARTH_RADIUS * np.cos(lat_rad) / nx
        dx = np.broadcast_to(dx[:, None], (ny, nx))
        
        # Zonal Sum: v * dx -> (z, y)
        transport = np.sum(v * dx[None, :, :], axis=2) 
        
        # Vertical Integration (Top Down) -> Psi(z)
        # Psi(z) = - Integral_z^0 v dz 
        moc = -np.cumsum(transport * dz, axis=0) # (z, y)
        moc_sv = moc / 1.0e6
        
        # Target Index at ~30N
        # (30 - (-90)) / 180 * ny = 120/180 * ny = 2/3 * ny
        target_idx = int((2/3) * ny)
        target_idx = min(target_idx, ny-1)
        
        # Max overturning in depth at this lat
        # Typically we look for the max positive cell in the upper 2000m or so
        amoc_val = np.max(moc_sv[:, target_idx])
        return amoc_val

    # 2. Loop through files for Timeseries
    for i, f in enumerate(files):
        # Use h5netcdf engine as netcdf4 had issues
        try:
            with xr.open_dataset(f, decode_times=False, engine='netcdf4') as ds:
                # Global Mean Temp
                # Check for temp variable name
                if 'atmos_temp' in ds:
                    t_var = ds['atmos_temp']
                elif 'temp' in ds:
                    t_var = ds['temp']
                else:
                    t_var = None
                
                if t_var is not None:
                    # Simple weighted mean
                    # Assuming basic linear latitude grid if lat not present
                    if 'lat' in ds:
                        lat = ds.lat
                    else:
                        nlat = t_var.shape[-2]
                        lat = xr.DataArray(np.linspace(-90, 90, nlat), dims='lat')
                    
                    weights = np.cos(np.deg2rad(lat))
                    t_glob = t_var.weighted(weights).mean().values
                    t_mean_list.append(t_glob)
                else:
                    t_mean_list.append(np.nan)

                # AMOC
                amoc_val = compute_amoc_point(ds)
                amoc_list.append(amoc_val)
                
                if i % 12 == 0:
                    print(f"Processed Month {i+1}/{len(files)}")

        except Exception as e:
            print(f"Error processing {f}: {e}")
            t_mean_list.append(np.nan)
            amoc_list.append(np.nan)


    # 3. Plot Timeseries
    # AMOC
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(amoc_list)+1), amoc_list, label='AMOC Index (30N)')
    plt.axhline(y=17, color='r', linestyle='--', alpha=0.5, label='Target (17 Sv)')
    plt.xlabel('Months')
    plt.ylabel('AMOC (Sv)')
    plt.title(f'Tuned Run AMOC Strength (Months 1-{len(files)})')
    plt.legend()
    plt.grid(True)
    plt.savefig('tuned_amoc_timeseries.png')
    print("Saved tuned_amoc_timeseries.png")

    # Temp
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(t_mean_list)+1), t_mean_list, label='Global Mean T_air')
    plt.axhline(y=288, color='g', linestyle='--', alpha=0.5, label='Target (288 K)')
    plt.xlabel('Months')
    plt.ylabel('Temperature (K)')
    plt.title(f'Tuned Run Global Temperature (Months 1-{len(files)})')
    plt.legend()
    plt.grid(True)
    plt.savefig('tuned_t_mean_timeseries.png')
    print("Saved tuned_t_mean_timeseries.png")
    
    print(f"Final AMOC (Month {len(files)}): {amoc_list[-1]:.2f} Sv")
    print(f"Final T_mean (Month {len(files)}): {t_mean_list[-1]:.2f} K")


    # 4. Snapshot / Climatology Maps (Last 12 months)
    # Using last month for simplicity if run is short, or mean of last year
    last_files = files[-12:]
    print(f"Generating maps from last {len(last_files)} months...")
    
    try:
        # ds_clim = xr.open_mfdataset(last_files, decode_times=False, engine='netcdf4', combine='by_coords').mean(dim='time')
        # Manual load to avoid dask
        ds_list = []
        for f in last_files:
            ds_list.append(xr.open_dataset(f, decode_times=False, engine='netcdf4'))
        ds_clim = xr.concat(ds_list, dim='time').mean(dim='time')
        
        # Load Land Mask for cleaner plots
        mask_land = np.array(data.load_bathymetry_mask())
        
        # SST Map
        if 'sst' in ds_clim or 'ocean_temp' in ds_clim:
            plt.figure(figsize=(10, 6))
            if 'sst' in ds_clim:
                sst = ds_clim['sst'].values
            else:
                sst = ds_clim['ocean_temp'].isel(z=0).values # Top layer
                
            # Mask Land
            sst_masked = np.ma.masked_where(~mask_land, sst)
            
            plt.imshow(sst_masked, origin='lower', cmap='RdYlBu_r', vmin=270, vmax=305)
            plt.colorbar(label='SST (K)')
            plt.title('Sea Surface Temperature (Last 12-Month Mean)')
            plt.savefig('tuned_sst_map.png')
            print("Saved tuned_sst_map.png")
        
        # SSS Map
        if 'ocean_salt' in ds_clim:
            plt.figure(figsize=(10, 6))
            sss = ds_clim['ocean_salt'].isel(z=0).values
            sss_masked = np.ma.masked_where(~mask_land, sss)
            
            plt.imshow(sss_masked, origin='lower', cmap='viridis')
            plt.colorbar(label='Salinity (psu)')
            plt.title('Sea Surface Salinity (Last 12-Month Mean)')
            plt.savefig('tuned_sss_map.png')
            print("Saved tuned_sss_map.png")
            
        # Coupling Diff (SST - Air Temp)
        # Need to align grids if necessary, usually they match in this model
        if 'atmos_temp' in ds_clim and ('sst' in ds_clim or 'ocean_temp' in ds_clim):
            t_air = ds_clim['atmos_temp'].values
            if 'sst' in ds_clim:
                t_ocean = ds_clim['sst'].values
            else:
                t_ocean = ds_clim['ocean_temp'].isel(z=0).values
            
            diff = t_ocean - t_air
            # Mask Land
            diff_masked = np.ma.masked_where(~mask_land, diff)
            
            plt.figure(figsize=(10, 6))
            plt.imshow(diff_masked, origin='lower', cmap='RdBu_r', vmin=-10, vmax=10)
            plt.colorbar(label='SST - Air Temp (K)')
            plt.title('Coupling Difference (SST - SAT) (Last 12-Month Mean)')
            plt.savefig('tuned_coupling_diff.png')
            print("Saved tuned_coupling_diff.png")

    except Exception as e:
        print(f"Error generating maps: {e}")

if __name__ == "__main__":
    analyze_tuned_run()
