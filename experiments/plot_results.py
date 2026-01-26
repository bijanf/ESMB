
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def plot_control_run():
    # Load all monthly means
    files = sorted(glob.glob("outputs/control_run/mean_*.nc"))
    if not files:
        print("No output files found!")
        return
        
    print(f"Loading {len(files)} files...")
    
    t_mean_list = []
    t_nh_list = []
    
    # Manual Loop to avoid dask requirement for open_mfdataset
    for f in files:
        # Decode times=False for safety
        with xr.open_dataset(f, decode_times=False, engine='h5netcdf') as ds:
            # print("Keys:", list(ds.keys())) # Debug if needed
            
            # Identify Lat coordinate
            if 'lat' in ds:
                lat = ds.lat
            elif 'y_atm' in ds:
                # Assume T31/T63 gaussian grid or similar. 
                # Need values.
                lat = ds.y_atm
                # If dimensionless index, we need actual lat values.
                # Usually saved as coordinate or variable.
            elif 'latitude' in ds:
                lat = ds.latitude
            else:
                # Fallback: Assume linear -90 to 90
                nlat = ds['temp'].shape[-2]
                lat = np.linspace(-90, 90, nlat)
                lat = xr.DataArray(lat, dims='lat')
            
            # Calculate Global Mean (weighted)
            weights = np.cos(np.deg2rad(lat))
            
            # Check for temp variable
            if 'atmos_temp' in ds:
                t_var = ds['atmos_temp']
            elif 'temp' in ds:
                t_var = ds['temp']
            else:
                print(f"No temp variable found in {f}")
                continue
                
            t_glob = t_var.weighted(weights).mean().values
            
            # NH Mean
            # Use 'lat' or 'y_atm' coordinate for filtering
            # If lat is DataArray, we can use where
            # Since lat is now xr.DataArray (from previous block), we must ensure it aligns with t_var dims
            if t_var.shape == lat.shape:
               is_nh = lat > 0
            elif t_var.shape[-2] == lat.shape[0]: # assume lat is 1D (ny)
               # Create boolean mask
               is_nh = ds.lat > 0 if 'lat' in ds else lat > 0
            
            # Simplified: Use numpy slicing if grid is known (NH is top half)
            ny = t_var.shape[-2]
            t_nh_val = t_var[..., ny//2:, :].mean().values # Simple top-half mean
            
            t_mean_list.append(t_glob)
            t_nh_list.append(t_nh_val)
            
    t_mean = np.array(t_mean_list)
    t_nh = np.array(t_nh_list)
    
    # Plot Timeseries
    plt.figure(figsize=(10, 5))
    plt.plot(t_mean, label='Global Mean T_air')
    plt.plot(t_nh, label='NH Mean T_air', linestyle='--')
    plt.xlabel('Months')
    plt.ylabel('Temperature (K)')
    plt.title(f'Control Run Progress (Months 0-{len(files)})')
    plt.legend()
    plt.grid(True)
    plt.savefig('t_mean_timeseries.png')
    print("Saved t_mean_timeseries.png")
    
    plt.savefig('t_mean_timeseries.png')
    print("Saved t_mean_timeseries.png")
    
    # Snapshot Analysis (Last Month)
    last_file = files[-1]
    last_ds = xr.open_dataset(last_file, decode_times=False, engine='h5netcdf')
    
    # 2. Snow Depth (Check for Winter Accumulation)
    # 2. Snow Depth (Check for Winter Accumulation)
    if 'snow_depth' in last_ds:
        plt.figure(figsize=(10, 6))
        snow = last_ds['snow_depth'].values
        # Mask near zero for visualization
        snow = np.ma.masked_where(snow < 0.001, snow)
        plt.imshow(snow, origin='lower', vmin=0, vmax=0.5, cmap='PuBuGn')
        plt.colorbar(label='Snow Depth (m)')
        plt.title(f'Snow Depth (m) - Month {len(files)} (Snapshot)')
        plt.savefig('snow_depth_map.png')
        print("Saved snow_depth_map.png")
        print(f"Max Snow Depth: {np.max(last_ds['snow_depth']):.4f} m")
    elif 'land_snow_depth' in last_ds: # Possible alternative name
        plt.figure(figsize=(10, 6))
        snow = last_ds['land_snow_depth'].values
        snow = np.ma.masked_where(snow < 0.001, snow)
        plt.imshow(snow, origin='lower', vmin=0, vmax=0.5, cmap='PuBuGn')
        plt.colorbar(label='Snow Depth (m)')
        plt.title(f'Snow Depth (m) - Month {len(files)} (Snapshot)')
        plt.savefig('snow_depth_map.png')
        print("Saved snow_depth_map.png")
    else:
        print("Variable 'snow_depth' not found.")
        # Debug: Print all keys
        print("Keys available:", list(last_ds.keys()))

    # 3. Coupling Check (Surf - Atmos Temp)
    # Check variables for Surface Temp
    t_surf = None
    if 'sst' in last_ds:
        t_surf = last_ds['sst'].values
    elif 'ocean_temp' in last_ds: 
        ot = last_ds['ocean_temp'].values
        if ot.ndim == 3: t_surf = ot[0]
        elif ot.ndim == 2: t_surf = ot
        else: t_surf = ot
    
    # Try getting Land Temp if available
    if 'land_temp' in last_ds:
        # If we have land temp, we can make a composite surface temp map
        # But for now, ocean temp is a good proxy for most of the globe
        pass

    t_atoms = None
    if 'atmos_temp' in last_ds:
        t_atoms = last_ds['atmos_temp'].values
    elif 'temp' in last_ds:
        t_atoms = last_ds['temp'].values
    
    if t_surf is not None and t_atoms is not None:
        # Ensure shapes match
        if t_surf.shape == t_atoms.shape:
             pass
        elif t_surf.shape == t_atoms.shape[-2:]:
             t_atoms = t_atoms[0] if t_atoms.ndim == 3 else t_atoms
        
        diff = t_surf - t_atoms
        
        plt.figure(figsize=(10, 6))
        plt.imshow(diff, origin='lower', vmin=-20, vmax=20, cmap='RdBu_r')
        plt.colorbar(label='Surface - Air Temp (K)')
        plt.title('Surface - Air Temp Difference (K) (Snapshot)')
        plt.savefig('coupling_diff_map.png')
        print("Saved coupling_diff_map.png")
        print(f"Coupling Max Diff: {np.max(diff):.2f} K")
        print(f"Coupling Min Diff: {np.min(diff):.2f} K")

if __name__ == "__main__":
    plot_control_run()
