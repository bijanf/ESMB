
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

def analyze_final_run():
    # 1. Setup paths and output files
    input_pattern = "outputs/control_run_final/mean_*.nc"
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
        if v.ndim == 4: v = v[0] 
        
        nz, ny, nx = v.shape
        dz = 5000.0 / nz
        lat_vals = np.linspace(-90, 90, ny)
        lat_rad = np.deg2rad(lat_vals)
        dx = 2 * np.pi * EARTH_RADIUS * np.cos(lat_rad) / nx
        dx = np.broadcast_to(dx[:, None], (ny, nx))
        
        transport = np.sum(v * dx[None, :, :], axis=2) 
        moc = -np.cumsum(transport * dz, axis=0) # (z, y)
        moc_sv = moc / 1.0e6
        
        target_idx = int((2/3) * ny)
        target_idx = min(target_idx, ny-1)
        
        amoc_val = np.max(moc_sv[:, target_idx])
        return amoc_val

    # 2. Loop through files for Timeseries
    for i, f in enumerate(files):
        try:
            with xr.open_dataset(f, decode_times=False, engine='netcdf4') as ds:
                # Temp
                if 'atmos_temp' in ds:
                    t_var = ds['atmos_temp']
                elif 'temp' in ds:
                    t_var = ds['temp']
                else:
                    t_var = None
                
                if t_var is not None:
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
    plt.title(f'Final Run AMOC Strength (Months 1-{len(files)})')
    plt.legend()
    plt.grid(True)
    plt.savefig('final_amoc_timeseries.png')
    print("Saved final_amoc_timeseries.png")

    # Temp
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(t_mean_list)+1), t_mean_list, label='Global Mean T_air')
    plt.axhline(y=288, color='g', linestyle='--', alpha=0.5, label='Target (288 K)')
    plt.xlabel('Months')
    plt.ylabel('Temperature (K)')
    plt.title(f'Final Run Global Temperature (Months 1-{len(files)})')
    plt.legend()
    plt.grid(True)
    plt.savefig('final_t_mean_timeseries.png')
    print("Saved final_t_mean_timeseries.png")
    
    if amoc_list:
        print(f"Final AMOC (Month {len(files)}): {amoc_list[-1]:.2f} Sv")
        print(f"Mean AMOC (Last 12 Months): {np.mean(amoc_list[-12:]):.2f} Sv")
    if t_mean_list:
        print(f"Final T_mean (Month {len(files)}): {t_mean_list[-1]:.2f} K")


    # 4. Snapshot / Climatology Maps (Last 12 months using Manual Mean)
    last_files = files[-12:] if len(files) >= 12 else files
    print(f"Generating maps from last {len(last_files)} months (Manual Mean)...")
    
    try:
        # Manual climatology to avoid dask
        ds_clim = None
        count = 0
        for f in last_files:
            ds = xr.open_dataset(f, decode_times=False, engine='netcdf4')
            if ds_clim is None:
                ds_clim = ds
            else:
                ds_clim = ds_clim + ds 
            count += 1
        
        ds_clim = ds_clim / count
        
        # Load Land Mask
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
            plt.savefig('final_sst_map.png')
            print("Saved final_sst_map.png")
            
        # SAT Map (Surface Air Temperature)
        if 'atmos_temp' in ds_clim:
            plt.figure(figsize=(10, 6))
            sat = ds_clim['atmos_temp'].values
            # If 3D, take lowest level (closest to surface) or if it's 2D use as is
            if sat.ndim == 3:
                sat = sat[-1] # Usually bottom level is last index in many models, check if needed
            
            plt.imshow(sat, origin='lower', cmap='RdYlBu_r', vmin=250, vmax=310)
            plt.colorbar(label='SAT (K)')
            plt.title('Surface Air Temperature (Last 12-Month Mean)')
            plt.savefig('final_sat_map.png')
            print("Saved final_sat_map.png")


    except Exception as e:
        print(f"Error generating maps: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_final_run()
