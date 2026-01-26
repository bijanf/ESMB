import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import sys
import argparse
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator

# Ensure we can import chronos_esm components if needed
sys.path.append(os.getcwd())

def open_ds_robust(path):
    """Try multiple engines to open a NetCDF file."""
    engines = [None, "h5netcdf", "scipy", "netcdf4"]
    for eng in engines:
        try:
            ds = xr.open_dataset(path, decode_times=False, engine=eng)
            return ds
        except Exception:
            continue
    raise ValueError(f"Could not open {path} with any engine.")

def load_mask_manual():
    """Load WOA18 mask manually to bypass JAX/Dependency issues."""
    try:
        # Locate File
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "chronos_esm")
        path_t = os.path.join(cache_dir, "woa18_decav_t00_5d.nc")
        
        if not os.path.exists(path_t):
            print(f"[WARN] Mask file not found at {path_t}")
            return None
            
        # Load WOA
        ds_t = open_ds_robust(path_t)
        
        # Create Mask (Valid Temp = Ocean)
        if 't_mn' in ds_t:
            temp_surf = ds_t.t_mn.isel(time=0, depth=0)
        elif 't_an' in ds_t:
            temp_surf = ds_t.t_an.isel(time=0, depth=0)
        else:
            print("[WARN] Unknown variable in WOA file for mask derivation.")
            return None
            
        mask_woa = ~np.isnan(temp_surf.values)
        
        # Regrid to 96x48 (T31)
        lat_woa = ds_t.lat.values
        lon_woa = ds_t.lon.values
        
        interp = RegularGridInterpolator(
            (lat_woa, lon_woa),
            mask_woa.astype(float),
            bounds_error=False,
            fill_value=0.0,
            method="nearest"
        )
        
        # Target Grid (T31)
        lat_model = np.linspace(-90, 90, 48)
        lon_model = np.linspace(-180, 180, 96)
        
        Y, X = np.meshgrid(lat_model, lon_model, indexing="ij")
        pts = np.array([Y.ravel(), X.ravel()]).T
        
        mask_interp = interp(pts).reshape(48, 96)
        
        return mask_interp > 0.5
        
    except Exception as e:
        print(f"[WARN] Manual mask loading failed: {e}")
        return None

def analyze_run(run_dir, label):
    """Analyze a specific run directory."""
    DATA_DIR = Path(run_dir)
    OUTPUT_DIR = Path("analysis_results") / label
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing {DATA_DIR} -> {OUTPUT_DIR}")
    
    # 1. Load Data
    files = sorted(glob.glob(str(DATA_DIR / "mean_*.nc")))
    if not files:
        print("No mean_*.nc files found.")
        return

    print(f"Found {len(files)} files. Loading data...")
    
    # Load Weights (Cos(lat))
    lat = np.linspace(-90, 90, 48)
    weights = np.cos(np.deg2rad(lat))
    weights /= weights.mean()
    
    # --- Time Series ---
    ts_data = {
        'months': [],
        'global_sat': [],
        'global_sst': [],
        'global_sss': [],
        'nino34': []
    }
    
    # Nino3.4 Indices (170W-120W, 5S-5N)
    lon_nino = np.linspace(-180, 180, 96)
    lat_nino = np.linspace(-90, 90, 48)
    idx_lat_nino = np.where((lat_nino >= -5) & (lat_nino <= 5))[0]
    idx_lon_nino = np.where((lon_nino >= -170) & (lon_nino <= -120))[0]

    for i, f in enumerate(files):
        try:
            with open_ds_robust(f) as ds:
                ts_data['months'].append(i+1)
                
                # SAT
                if 'atmos_temp' in ds:
                    sat = ds['atmos_temp'].values
                    if sat.ndim == 3: sat = sat[0] # Surface
                    sat_mean = np.average(sat, axis=(0), weights=weights) if sat.shape[0] == 48 else sat.mean()
                    # Axis 0 is lat for (lat,lon)
                    ts_data['global_sat'].append(sat_mean)
                else:
                    ts_data['global_sat'].append(np.nan)
                    
                # SST (Use Ocean Temp Layer 0)
                sst = None
                if 'ocean_temp' in ds:
                    sst = ds['ocean_temp'][0].values
                elif 'sst' in ds:
                    sst = ds['sst'].values
                
                if sst is not None:
                     sst_mean = np.average(sst, axis=0, weights=weights).mean() # Approximate global mean
                     ts_data['global_sst'].append(sst_mean)
                     
                     # Nino3.4
                     if len(idx_lat_nino) > 0 and len(idx_lon_nino) > 0:
                         sst_nino = sst[np.ix_(idx_lat_nino, idx_lon_nino)]
                         ts_data['nino34'].append(sst_nino.mean())
                     else:
                        ts_data['nino34'].append(np.nan)
                else:
                    ts_data['global_sst'].append(np.nan)
                    ts_data['nino34'].append(np.nan)
                    
                # SSS
                if 'ocean_salt' in ds:
                    sss = ds['ocean_salt'][0].values
                    sss_mean = sss.mean()
                    ts_data['global_sss'].append(sss_mean)
                else:
                    ts_data['global_sss'].append(np.nan)

        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    # Plot Time Series
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Temperature (K)', color='tab:red')
    ax1.plot(ts_data['months'], ts_data['global_sat'], color='tab:red', label='Global SAT', linestyle='-')
    ax1.plot(ts_data['months'], ts_data['global_sst'], color='tab:orange', label='Global SST', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Nino3.4 Index (K)', color='tab:brown')
    ax2.plot(ts_data['months'], ts_data['nino34'], color='tab:brown', label='Nino3.4', linestyle=':')
    ax2.tick_params(axis='y', labelcolor='tab:brown')
    
    plt.title(f'Time Series: {label}')
    fig.tight_layout()
    plt.savefig(OUTPUT_DIR / "timeseries_summary.png")
    
    # --- Climatology Maps (Last 12 Months) ---
    print("Generating Climatology Maps...")
    last_files = files[-12:] if len(files) >= 12 else files
    
    # Load mask
    mask_ocean = load_mask_manual()
    mask_land = ~mask_ocean if mask_ocean is not None else None
    
    if not last_files:
        print("No files for climatology.")
        return

    # Aggregate
    sat_sum = None
    sst_sum = None
    count = 0
    
    for f in last_files:
        with open_ds_robust(f) as ds:
            # SAT
            if 'atmos_temp' in ds:
                val = ds['atmos_temp'].values
                if val.ndim == 3: val = val[0]
                sat_sum = val if sat_sum is None else sat_sum + val
            
            # SST
            val_sst = None
            if 'ocean_temp' in ds:
                val_sst = ds['ocean_temp'][0].values
            elif 'sst' in ds:
                val_sst = ds['sst'].values
            
            if val_sst is not None:
                sst_sum = val_sst if sst_sum is None else sst_sum + val_sst
                
            count += 1
            
    if count > 0:
        # SAT Map
        if sat_sum is not None:
            sat_clim = sat_sum / count
            plt.figure(figsize=(10, 6))
            plt.imshow(sat_clim, origin='lower', cmap='RdBu_r', vmin=250, vmax=310)
            plt.colorbar(label='K')
            plt.title('SAT Climatology (Last 12 Months)')
            plt.savefig(OUTPUT_DIR / "map_sat.png")
            plt.close()
            
        # SST Map (Masked)
        if sst_sum is not None:
            sst_clim = sst_sum / count
            
            if mask_land is not None:
                sst_clim = np.ma.masked_where(mask_land, sst_clim)
                
            plt.figure(figsize=(10, 6))
            plt.imshow(sst_clim, origin='lower', cmap='RdBu_r', vmin=270, vmax=305)
            plt.colorbar(label='K')
            plt.title('SST Climatology (Masked)')
            plt.savefig(OUTPUT_DIR / "map_sst.png")
            plt.close()
            
        # Coupling Diff
        if sat_sum is not None and sst_sum is not None:
             sat_clim = sat_sum / count
             sst_clim = sst_sum / count # Raw
             
             diff = sat_clim - sst_clim
             if mask_land is not None:
                 diff = np.ma.masked_where(mask_land, diff)
                 
             plt.figure(figsize=(10, 6))
             plt.imshow(diff, origin='lower', cmap='RdBu_r', vmin=-5, vmax=5)
             plt.colorbar(label='K')
             plt.title('Coupling Diff: SAT - SST')
             plt.savefig(OUTPUT_DIR / "map_coupling_diff.png")
             plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Chronos-ESM Run")
    parser.add_argument("--dir", type=str, required=True, help="Path to run output directory")
    parser.add_argument("--label", type=str, required=True, help="Label for analysis output")
    args = parser.parse_args()
    
    analyze_run(args.dir, args.label)
