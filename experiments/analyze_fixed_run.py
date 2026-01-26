
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
try:
    from chronos_esm import data
except:
    pass

def analyze_fixed_run():
    # 1. Setup paths and output files
    input_pattern = "outputs/control_run_fixed/mean_*.nc"
    files = sorted(glob.glob(input_pattern))
    
    if not files:
        print(f"No files found matching {input_pattern}")
        return

    print(f"Found {len(files)} output files. Starting analysis...")

    # Lists to store timeseries data
    t_mean_list = []
    
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
                    # Simple mean if lat not weighted, but let's just do simple for speed check
                    t_glob = t_var.mean().values
                    t_mean_list.append(t_glob)
                else:
                    t_mean_list.append(np.nan)
                    
        except Exception as e:
            print(f"Error processing {f}: {e}")
            t_mean_list.append(np.nan)

    # 3. Plot Timeseries
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(t_mean_list)+1), t_mean_list, label='Global Mean T_air')
    plt.axhline(y=288, color='g', linestyle='--', alpha=0.5, label='Target (288 K)')
    plt.xlabel('Months')
    plt.ylabel('Temperature (K)')
    plt.title(f'Fixed Run Global Temperature (Months 1-{len(files)})')
    plt.legend()
    plt.grid(True)
    plt.savefig('fixed_t_mean_timeseries.png')
    print("Saved fixed_t_mean_timeseries.png")
    
    if t_mean_list:
        print(f"Final T_mean (Month {len(files)}): {t_mean_list[-1]:.2f} K")


    # 4. Snapshot Maps (Last Month)
    last_file = files[-1]
    print(f"Generating maps from last month: {last_file}")
    
    try:
        ds = xr.open_dataset(last_file, decode_times=False, engine='netcdf4')
        
        # Load Land Mask
        try:
            mask_land = np.array(data.load_bathymetry_mask())
        except:
             mask_land = np.zeros((48, 96), dtype=bool) # Fallback

        # SST Map
        if 'sst' in ds or 'ocean_temp' in ds:
            plt.figure(figsize=(10, 6))
            if 'sst' in ds:
                sst = ds['sst'].values
            else:
                sst = ds['ocean_temp'].isel(z=0).values # Top layer
                
            # Mask Land
            sst_masked = np.ma.masked_where(~mask_land, sst)
            
            plt.imshow(sst_masked, origin='lower', cmap='RdYlBu_r', vmin=260, vmax=310)
            plt.colorbar(label='SST (K)')
            plt.title(f'SST (Month {len(files)})')
            plt.savefig('fixed_sst_map.png')
            print("Saved fixed_sst_map.png")
            
        # SAT Map
        if 'atmos_temp' in ds:
            plt.figure(figsize=(10, 6))
            sat = ds['atmos_temp'].values
            plt.imshow(sat, origin='lower', cmap='RdYlBu_r', vmin=260, vmax=310)
            plt.colorbar(label='SAT (K)')
            plt.title(f'SAT (Month {len(files)})')
            plt.savefig('fixed_sat_map.png')
            print("Saved fixed_sat_map.png")

    except Exception as e:
        print(f"Error generating maps: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_fixed_run()
