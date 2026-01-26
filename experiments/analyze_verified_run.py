
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys
from pathlib import Path

# Ensure we can import chronos_esm
sys.path.append(os.getcwd())

def analyze_verified_run():
    # Configuration
    DATA_DIR = Path("outputs/control_run_verified")
    OUTPUT_DIR = Path("analysis_results/verified_run")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all daily/monthly mean files
    files = sorted(list(DATA_DIR.glob("mean_*.nc")))
    if not files:
        print(f"No output files found in {DATA_DIR}")
        return

    print(f"Found {len(files)} files. Loading data...")
    
    # Helper to open dataset robustly
    def open_ds_robust(path):
        for eng in [None, "h5netcdf", "scipy", "netcdf4"]:
            try:
                ds = xr.open_dataset(path, decode_times=False, engine=eng)
                return ds
            except Exception:
                continue
        raise ValueError(f"Could not open {path}")

    # --- 1. Time-Series Analysis ---
    print("Generating Time-Series...")
    
    ts_data = {
        'time': [],
        'amoc_30n': [],
        'amoc_max': [],
        'global_sat': [], # Surface Air Temp
        'global_sst': [],
        'global_sss': [],
        'toa_balance': []
    }
    
    # Geometry Helper (assume regular grid if not present)
    # T31 approx: 96x48
    nx, ny = 96, 48
    R_EARTH = 6.371e6
    
    # Helper for manual processing if mfdataset fails or for complex derived diagnostics
    for i, f in enumerate(files):
        if i % 12 == 0: print(f"Processing Year {i//12}...")
        
        try:
            with open_ds_robust(f) as d_step:
                # Time
                ts_data['time'].append(i)
                
                # --- Global Mean Temps ---
                # Weights
                if 'lat' in d_step:
                    lat = d_step.lat
                else:
                    lat = np.linspace(-90, 90, ny)
                weights = np.cos(np.deg2rad(lat))
                
                # SAT
                if 'atmos_temp' in d_step:
                    sat = d_step['atmos_temp']
                    if sat.ndim == 3: sat = sat.mean(dim=sat.dims[-1]) 
                    # Assuming (lat, lon) or (y, x)
                    sat_mean = sat.weighted(xr.DataArray(weights, dims=sat.dims[0])).mean().item()
                    ts_data['global_sat'].append(sat_mean)
                else: 
                     ts_data['global_sat'].append(np.nan)

                # SST & SSS
                # NOTE: 'sst' variable in output might be masked or land-filled differently. 
                # Use ocean_temp[0] as true model SST.
                if 'ocean_temp' in d_step:
                    sst = d_step['ocean_temp'][0] if d_step['ocean_temp'].ndim==3 else d_step['ocean_temp']
                    sst_mean = sst.weighted(xr.DataArray(weights, dims=sst.dims[0])).mean().item()
                    ts_data['global_sst'].append(sst_mean)
                elif 'sst' in d_step: # Fallback
                    sst = d_step['sst']
                    sst_mean = sst.weighted(xr.DataArray(weights, dims=sst.dims[0])).mean().item()
                    ts_data['global_sst'].append(sst_mean)
                else:
                    ts_data['global_sst'].append(np.nan)
                    
                # SALINITY Check
                if 'ocean_salt' in d_step:
                    sss = d_step['ocean_salt'][0] if d_step['ocean_salt'].ndim==3 else d_step['ocean_salt']
                    sss_mean = sss.weighted(xr.DataArray(weights, dims=sss.dims[0])).mean().item()
                    ts_data['global_sss'].append(sss_mean)
                elif 'salinity' in d_step:
                    sss = d_step['salinity'][0] if d_step['salinity'].ndim==3 else d_step['salinity']
                    sss_mean = sss.weighted(xr.DataArray(weights, dims=sss.dims[0])).mean().item()
                    ts_data['global_sss'].append(sss_mean)
                else:
                    ts_data['global_sss'].append(np.nan)

                # --- AMOC ---
                if 'ocean_v' in d_step:
                    v = d_step['ocean_v'].values
                    if v.ndim == 4: v = v[0] # time, z, y, x
                    
                    # Zonal Integrals
                    dx = 2 * np.pi * R_EARTH * np.cos(np.deg2rad(lat)) / nx
                    v_zonal = np.sum(v * dx[:, None] if dx.ndim==1 else dx, axis=2) # (z, y)
                    
                    # Vertical Integral (from top)
                    dz = np.array([50.0, 50.0, 100.0, 100.0, 100.0, 200.0, 200.0, 300.0, 
                                   400.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0])
                    
                    psi = -np.cumsum(v_zonal * dz[:, None], axis=0) / 1e6 
                    
                    amoc_all = np.max(np.abs(psi))
                    ts_data['amoc_max'].append(amoc_all)
                    
                    # 30N index (approx index 32 for 96x48)
                    idx_30n = int((30+90)/180 * ny)
                    ts_data['amoc_30n'].append(np.max(psi[:, idx_30n]))
                else:
                    ts_data['amoc_max'].append(np.nan)
                    ts_data['amoc_30n'].append(np.nan)
                    
        except Exception as e:
            print(f"Error processing {f.name}: {e}")
            
    # Plot Time-Series
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Temperature
    axes[0].plot(ts_data['time'], ts_data['global_sat'], label='SAT (K)', color='tab:red')
    axes[0].plot(ts_data['time'], ts_data['global_sst'], label='SST (K)', color='tab:blue')
    axes[0].set_title('Global Mean Temperature')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. AMOC
    axes[1].plot(ts_data['time'], ts_data['amoc_30n'], label='AMOC @ 30N (Sv)', color='black')
    axes[1].plot(ts_data['time'], ts_data['amoc_max'], label='Global Max Overturning (Sv)', color='gray', linestyle='--')
    axes[1].set_title('Overturning Strength')
    axes[1].set_ylabel('Sv')
    axes[1].legend()
    axes[1].grid(True)
    
    # 3. SSS
    axes[2].plot(ts_data['time'], ts_data['global_sss'], label='Global Mean SSS (g/kg)', color='tab:green')
    axes[2].set_title('Global Mean Salinity')
    axes[2].set_xlabel('Months')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "timeseries_main.png")
    print("Saved timeseries_main.png")

    # --- 2. Climatology Maps (Last 12 Months) ---
    print("Generating Climatology Maps...")
    
    last_n_files = files[-12:]
    if not last_n_files: return
    
    # Manual Mean if mfdataset fails
    try:
        ds_clim = None
        count = 0
        for f in last_n_files:
            try:
                d = open_ds_robust(f)
                if ds_clim is None:
                    ds_clim = d
                else:
                    ds_clim = ds_clim + d
                count += 1
            except:
                pass
        
        if ds_clim is not None:
            ds_clim = ds_clim / count
            print("Climatology computed manually.")
    except Exception as e:
        print(f"Failed climatology: {e}")
        return
    
    # A. Surface Air Temperature
    plt.figure(figsize=(10, 6))
    if 'atmos_temp' in ds_clim:
        plt.imshow(ds_clim['atmos_temp'], origin='lower', cmap='RdBu_r')
        plt.colorbar(label='K')
        plt.title('Surface Air Temperature (Last 12 Months Avg)')
        plt.savefig(OUTPUT_DIR / "map_sat_climatology.png")
    
    # Manual Mask Loading Helper
    def load_mask_manual():
        try:
            from scipy.interpolate import RegularGridInterpolator
            
            # 1. Locate File
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "chronos_esm")
            path_t = os.path.join(cache_dir, "woa18_decav_t00_5d.nc")
            
            if not os.path.exists(path_t):
                print(f"Mask file not found at {path_t}")
                return None
                
            # 2. Load WOA
            # Need robust opener again?
            ds_t = open_ds_robust(path_t)
            
            # 3. Create Mask (Valid Temp = Ocean)
            # dims: time, depth, lat, lon
            # t_mn or similar
            if 't_mn' in ds_t:
                temp_surf = ds_t.t_mn.isel(time=0, depth=0)
            elif 't_an' in ds_t:
                temp_surf = ds_t.t_an.isel(time=0, depth=0)
            else:
                print("Unknown variable in WOA file")
                return None
                
            mask_woa = ~np.isnan(temp_surf.values)
            
            # 4. Regrid to 96x48 (T31)
            lat_woa = ds_t.lat.values
            lon_woa = ds_t.lon.values
            
            # Create Interpolator
            interp = RegularGridInterpolator(
                (lat_woa, lon_woa),
                mask_woa.astype(float),
                bounds_error=False,
                fill_value=0.0,
                method="nearest"
            )
            
            # Target Grid
            lat_model = np.linspace(-90, 90, 48)
            lon_model = np.linspace(-180, 180, 96)
            
            Y, X = np.meshgrid(lat_model, lon_model, indexing="ij")
            pts = np.array([Y.ravel(), X.ravel()]).T
            
            mask_interp = interp(pts).reshape(48, 96)
            
            return mask_interp > 0.5
            
        except Exception as e:
            print(f"Manual mask loading failed: {e}")
            return None

    # Load Mask
    mask_land = None
    mask_ocean = load_mask_manual()
    if mask_ocean is not None:
        mask_land = ~mask_ocean
        print("Land mask loaded successfully (Manual).")
    else:
        print("Using unmasked plots.")
        
    # --- 3. ENSO (Nino3.4) Analysis ---
    # Nino3.4 Region: 5S-5N, 170W-120W
    # Model Lon: -180 to 180
    # 170W = -170, 120W = -120
    print("Generating Nino3.4 Analysis...")
    
    nino34_series = []
    
    for f in files:
        try:
            with open_ds_robust(f) as d_step:
                 # Get SST
                if 'ocean_temp' in d_step:
                    sst = d_step['ocean_temp'][0] if d_step['ocean_temp'].ndim==3 else d_step['ocean_temp']
                elif 'sst' in d_step:
                    sst = d_step['sst']
                else:
                    nino34_series.append(np.nan)
                    continue
                
                # Coords
                if 'lat' in d_step: lat = d_step.lat.values
                else: lat = np.linspace(-90, 90, 48)
                
                if 'lon' in d_step: lon = d_step.lon.values
                else: lon = np.linspace(-180, 180, 96)
                
                # Select Region
                # manual slice
                lat_idx = np.where((lat >= -5) & (lat <= 5))[0]
                lon_idx = np.where((lon >= -170) & (lon <= -120))[0]
                
                if len(lat_idx) > 0 and len(lon_idx) > 0:
                     # Subset
                     sst_region = sst.isel({sst.dims[0]: lat_idx, sst.dims[1]: lon_idx})
                     # Mean
                     nino34_series.append(sst_region.mean().item())
                else:
                     nino34_series.append(np.nan)
        except:
             nino34_series.append(np.nan)
             
    # Plot Nino3.4
    plt.figure(figsize=(10, 4))
    plt.plot(nino34_series, label='Nino3.4 Index (Raw SST)', color='tab:orange')
    plt.title('Nino3.4 Region SST (5S-5N, 170W-120W)')
    plt.ylabel('SST (K)')
    plt.xlabel('Months')
    plt.grid(True)
    plt.savefig(OUTPUT_DIR / "nino34_timeseries.png")
    print("Saved nino34_timeseries.png")
        
    # B. SST
    plt.figure(figsize=(10, 6))
    sst_val = None
    if 'ocean_temp' in ds_clim:
        sst_val = ds_clim['ocean_temp'][0].values
    elif 'sst' in ds_clim:
        sst_val = ds_clim['sst'].values
        
    if sst_val is not None:
        # Apply Mask if available
        if mask_land is not None:
            # Ensure shape match
            if sst_val.shape == mask_land.shape:
                sst_val = np.ma.masked_where(mask_land, sst_val)
            else:
                 print(f"Shape mismatch mask: {mask_land.shape} vs sst: {sst_val.shape}")
                 
        plt.imshow(sst_val, origin='lower', cmap='RdBu_r', vmin=270, vmax=305) # Better range
        plt.colorbar(label='K')
        plt.title('SST (Ocean Temp Layer 0) (Last 12 Months Avg)')
        plt.savefig(OUTPUT_DIR / "map_sst_climatology.png")

    # C. SSS
    plt.figure(figsize=(10, 6))
    sss_surf = None
    if 'ocean_salt' in ds_clim:
        sss_surf = ds_clim['ocean_salt'][0].values if ds_clim['ocean_salt'].ndim >= 3 else ds_clim['ocean_salt'].values
    elif 'salinity' in ds_clim:
        sss_surf = ds_clim['salinity'][0].values if ds_clim['salinity'].ndim == 3 else ds_clim['salinity'].values
    
    if sss_surf is not None:
         # Apply Mask if available
        if mask_land is not None and sss_surf.shape == mask_land.shape:
            sss_surf = np.ma.masked_where(mask_land, sss_surf)
            
        plt.imshow(sss_surf, origin='lower', cmap='viridis')
        plt.colorbar(label='g/kg')
        plt.title('SSS (Last 12 Months Avg)')
        plt.savefig(OUTPUT_DIR / "map_sss_climatology.png")
        
    # Coupling Difference
    if 'atmos_temp' in ds_clim and sst_val is not None:
        plt.figure(figsize=(10, 6))
        sat = ds_clim['atmos_temp']
        if sat.ndim == 3: sat = sat[sat.shape[0]//2] # Take middle or mean? 
        # Actually atmos_temp might be (time, y, x) but we did .mean(dim='time')
        # Check shapes
        # Force to numpy to avoid dim alignment issues
        sat_np = sat.values
        sst_np = sst_val # Already numpy from earlier block
        
        if sat_np.ndim == 3: sat_np = sat_np[0]
        if sst_np.ndim == 3: sst_np = sst_np[0]
        
        # Simple difference (assuming same grid)
        try:
            diff = sat_np - sst_np
            
            # Apply Mask to diff as well if available
            if mask_land is not None and diff.shape == mask_land.shape:
                diff = np.ma.masked_where(mask_land, diff)
                
            plt.imshow(diff, origin='lower', cmap='RdBu_r', vmin=-10, vmax=10)
            plt.colorbar(label='K')
            plt.title('SAT - SST Difference (Last 12 Months Avg)')
            plt.savefig(OUTPUT_DIR / "map_coupling_diff.png")
        except Exception as e:
            print(f"Coupling diff failed: {e}")
        
    # D. AMOC Streamfunction (Latitude vs Depth)
    if 'ocean_v' in ds_clim:
        v_clim = ds_clim['ocean_v'].values # (z, y, x)
        
        # Zonal Integrals
        # dx varies with lat
        ny_local = v_clim.shape[1]
        nx_local = v_clim.shape[2]
        lats_local = np.linspace(-90, 90, ny_local)
        dx_local = 2 * np.pi * R_EARTH * np.cos(np.deg2rad(lats_local)) / nx_local
        
        v_zonal_clim = np.sum(v_clim * dx_local[None, :, None], axis=2) # (z, y)
        
        # Psi
        dz = np.array([50.0, 50.0, 100.0, 100.0, 100.0, 200.0, 200.0, 300.0, 
                       400.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0])
                       
        psi_clim = -np.cumsum(v_zonal_clim * dz[:, None], axis=0) / 1e6
        
        plt.figure(figsize=(10, 6))
        # Contourf: Y-axis=Depth (flipped), X-axis=Latitude
        depths = np.cumsum(dz)
        depths_center = depths - dz/2
        
        X, Y = np.meshgrid(lats_local, depths_center)
        # Y is depth (positive down), so flip plot Y later
        
        levels = np.linspace(-25, 25, 51)
        plot_h = plt.contourf(X, Y, psi_clim, levels=levels, cmap='RdBu_r', extend='both')
        plt.colorbar(plot_h, label='Sv')
        plt.gca().invert_yaxis() # Surface at top
        plt.title('Meridional Overturning Circulation (Climatology)')
        plt.xlabel('Latitude')
        plt.ylabel('Depth (m)')
        plt.savefig(OUTPUT_DIR / "plot_amoc_structure.png")
        
    print("Analysis Complete.")

if __name__ == "__main__":
    analyze_verified_run()
