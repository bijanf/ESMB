
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def plot_production_run():
    # Load all monthly means
    files = sorted(glob.glob("outputs/production_control/mean_*.nc"))
    if not files:
        print("No output files found!")
        return
        
    print(f"Loading {len(files)} files...")
    
    t_mean_list = []
    amoc_list = []
    
    # Constants for AMOC
    EARTH_RADIUS = 6.371e6
    
    def compute_amoc_point(ds):
        if 'ocean_v' not in ds: return 0.0
        v = ds['ocean_v'].values
        if v.ndim == 4: v = v[0] # time, z, y, x -> z, y, x
        
        # Grid
        nz, ny, nx = v.shape
        dz = 5000.0 / nz
        
        # Lat
        if 'y_ocn' in ds: lat_idx = ds.y_ocn.values
        else: lat_idx = np.arange(ny)
        # Lat grid approximation: -90 to 90
        lat_vals = np.linspace(-90, 90, ny)
        
        # dx at each lat
        lat_rad = np.deg2rad(lat_vals)
        dx = 2 * np.pi * EARTH_RADIUS * np.cos(lat_rad) / nx
        dx = np.broadcast_to(dx[:, None], (ny, nx))
        
        # Zonal Sum: v * dx
        # v: (z, y, x), dx: (y, x) -> broadcast
        transport = np.sum(v * dx[None, :, :], axis=2) # (z, y)
        
        # Vertical Integration (Top Down)
        # Psi(z) = - Integral_z^0 v dz 
        # cumulative sum from top (low index)
        moc = -np.cumsum(transport * dz, axis=0)
        moc_sv = moc / 1.0e6
        
        # Index at 30N
        # 30N index: (30 - (-90))/180 * ny = 120/180 * 48 = 32
        target_idx = int(0.666 * ny)
        target_idx = min(target_idx, ny-1)
        
        # Max overturning in depth at this lat
        amoc_val = np.max(moc_sv[:, target_idx])
        return amoc_val

    # Process files
    for f in files:
        with xr.open_dataset(f, decode_times=False, engine='netcdf4') as ds:
            # Calculate Global Mean (weighted)
            if 'lat' in ds:
                lat = ds.lat
            else:
                 nlat = ds['temp'].shape[-2] if 'temp' in ds else ds['atmos_temp'].shape[-2]
                 lat = xr.DataArray(np.linspace(-90, 90, nlat), dims='lat')

            weights = np.cos(np.deg2rad(lat))
            
            if 'atmos_temp' in ds:
                t_var = ds['atmos_temp']
            elif 'temp' in ds:
                t_var = ds['temp']
            else:
                continue
                
            t_glob = t_var.weighted(weights).mean().values
            t_mean_list.append(t_glob)
            
            # AMOC
            amoc_list.append(compute_amoc_point(ds))
            
    # Plot AMOC Timeseries
    plt.figure(figsize=(10, 5))
    plt.plot(amoc_list, label='AMOC Index (30N)')
    plt.xlabel('Months')
    plt.ylabel('AMOC (Sv)')
    plt.title(f'AMOC Strength (Months 0-{len(files)})')
    plt.legend()
    plt.grid(True)
    plt.savefig('production_amoc_timeseries_v2.png')
    print("Saved production_amoc_timeseries_v2.png")
            
    # Plot Timeseries
    plt.figure(figsize=(10, 5))
    plt.plot(t_mean_list, label='Global Mean T_air')
    plt.xlabel('Months')
    plt.ylabel('Temperature (K)')
    plt.title(f'Production Control Run Progress (Months 0-{len(files)})')
    plt.legend()
    plt.grid(True)
    plt.savefig('production_t_mean_timeseries_v2.png')
    print("Saved production_t_mean_timeseries_v2.png")
    
    # Climatology Analysis (Last 30 Years / 360 Months)
    n_months = 360
    selected_files = files[-n_months:] if len(files) >= n_months else files
    print(f"Computing Climatology over last {len(selected_files)} months (Manual Loop)...")
    
    try:
        # Load first file to initialize sum
        ds_sum = xr.open_dataset(selected_files[0], decode_times=False, engine='netcdf4')
        for f in selected_files[1:]:
            with xr.open_dataset(f, decode_times=False, engine='netcdf4') as ds_next:
                 ds_sum += ds_next
        
        last_ds = ds_sum / len(selected_files)
        print("Climatology computed.")
    except Exception as e:
        print(f"Failed to compute manual climatology: {e}. Falling back to snapshot.")
        last_ds = xr.open_dataset(files[-1], decode_times=False, engine='netcdf4')

        
    # Coupling Check (Surf - Atmos Temp)
    # Load Mask
    import sys
    sys.path.append(os.getcwd())
    from chronos_esm import data
    mask_land = np.array(data.load_bathymetry_mask())

    t_surf = None
    if 'sst' in last_ds:
        t_surf = last_ds['sst'].values
    elif 'ocean_temp' in last_ds: 
        ot = last_ds['ocean_temp'].values
        if ot.ndim == 3: t_surf = ot[0] # Surface layer
        elif ot.ndim == 2: t_surf = ot
        else: t_surf = ot
        
    t_atoms = None
    if 'atmos_temp' in last_ds:
        t_atoms = last_ds['atmos_temp'].values
        
    # Load Ice Concentration for masking
    start_lines_to_view = 61
    mask_ice = None
    if 'ice_concentration' in last_ds:
        ice = last_ds['ice_concentration'].values
        if ice.ndim == 3: ice = ice[0]
        mask_ice = ice > 0.15 # Mask where ice is present
    
    if t_surf is not None and t_atoms is not None:
        if t_surf.shape == t_atoms.shape[-2:]: # Handle if atmos has levels or time
             t_atoms = t_atoms[0] if t_atoms.ndim == 3 else t_atoms
        
        # Simple resize if needed (very rough) - assuming grid matches for now
        if t_surf.shape == t_atoms.shape:
            diff = t_surf - t_atoms
            
            # Mask Land (mask_land=True is Ocean)
            diff = np.ma.masked_where(~mask_land, diff)
            
            # Mask Ice
            if mask_ice is not None:
                diff = np.ma.masked_where(mask_ice, diff) # Mask where ice is True
            
            plt.figure(figsize=(10, 6))
            plt.imshow(diff, origin='lower', vmin=-20, vmax=20, cmap='RdBu_r')
            plt.colorbar(label='Surface - Air Temp (K)')
            plt.title('Surface - Air Temp Difference (K) (Snapshot)')
            plt.savefig('production_coupling_diff_map_v2.png')
            print("Saved production_coupling_diff_map_v2.png")
            print(f"Coupling Max Diff: {np.max(diff):.2f} K")
            print(f"Coupling Min Diff: {np.min(diff):.2f} K")

if __name__ == "__main__":
    plot_production_run()
