
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import sys
import pooch
from scipy.interpolate import RegularGridInterpolator

# --- Standalone Data Loading (Avoids JAX import from chronos_esm) ---

def fetch_woa18_temp():
    """Download WOA18 5-degree annual temperature climatology."""
    fname_temp = "woa18_decav_t00_5d.nc"
    path_temp = pooch.retrieve(
        url="https://www.ncei.noaa.gov/thredds-ocean/fileServer/ncei/woa/temperature/decav/5deg/woa18_decav_t00_5d.nc",
        known_hash=None,
        path=pooch.os_cache("chronos_esm"),
        fname=fname_temp,
    )
    return path_temp

def load_bathymetry_mask_standalone(nlat=48, nlon=96):
    """
    Derive land mask from WOA18 temperature data.
    """
    path_t = fetch_woa18_temp()
    ds_t = xr.open_dataset(path_t, decode_times=False)

    # Load raw surface temp (time=0, depth=0)
    temp_surf = ds_t.t_mn.isel(time=0, depth=0)

    # Create mask: Valid data is Ocean
    mask_woa = ~np.isnan(temp_surf.values)

    # Regrid mask to model grid
    lat_woa = ds_t.lat.values
    lon_woa = ds_t.lon.values

    # T31 Grid
    lat_model = np.linspace(-90, 90, nlat)
    lon_model = np.linspace(-180, 180, nlon)

    # Interpolator
    interp = RegularGridInterpolator(
        (lat_woa, lon_woa),
        mask_woa.astype(float),
        bounds_error=False,
        fill_value=0.0,
        method="nearest",
    )

    Y, X = np.meshgrid(lat_model, lon_model, indexing="ij")
    pts = np.array([Y.ravel(), X.ravel()]).T

    mask_interp = interp(pts).reshape(nlat, nlon)
    mask_boolean = mask_interp > 0.5
    
    return mask_boolean

# --- Main Analysis Script ---

def analyze_prod_run():
    # Setup Paths
    output_dir = 'analysis_results/prod_v2'
    os.makedirs(output_dir, exist_ok=True)
    
    files = sorted(glob.glob('outputs/control_run_prod_v2/mean_*.nc'))
    if not files:
        print("No output files found!")
        return

    print(f"Found {len(files)} monthly files. Processing...")

    # --- Time Series Arrays ---
    times = []
    t_global_series = []
    sst_global_series = []
    amoc_series = []
    
    # Pre-compute Grid info for AMOC (T31 approx)
    ds_grid = xr.open_dataset(files[0], decode_times=False)
    nx = ds_grid.sizes['x_ocn']
    ny = ds_grid.sizes['y_ocn']
    nz = ds_grid.sizes['z']
    
    # Hardcoded Grid Config from analysis of config.py (T31)
    EARTH_RADIUS = 6.371e6
    dz = 5000.0 / nz
    lat_vals = np.linspace(-90, 90, ny)
    lat_rad = np.deg2rad(lat_vals)
    dx = 2 * np.pi * EARTH_RADIUS * np.cos(lat_rad) / nx
    dx_grid = np.broadcast_to(dx[:, None], (ny, nx)) # (ny, nx)
    
    # AMOC Index Range (20N - 60N)
    idx_20n = int(ny * (20+90)/180)
    idx_60n = int(ny * (60+90)/180)

    # --- Iterate for Time Series ---
    for i, f in enumerate(files):
        try:
            ds = xr.open_dataset(f, decode_times=False)
            
            # Time (Year)
            month = int(os.path.basename(f).split('_')[1].split('.')[0])
            year = month / 12.0
            times.append(year)
            
            # Global Mean SAT (Atmosphere)
            t_sat = ds.atmos_temp.mean().item()
            t_global_series.append(t_sat)
            
            # Global Mean SST (Ocean Layer 0)
            t_sst = ds.ocean_temp.isel(z=0).mean().item()
            sst_global_series.append(t_sst)
            
            # AMOC Calculation
            v = ds.ocean_v.values # (z, y, x)
            transport_zonal = np.sum(v * dx_grid[None, :, :], axis=2) # (z, y)
            moc = -np.cumsum(transport_zonal * dz, axis=0) # (z, y) Top-down integration
            moc_sv = moc / 1.0e6
            
            # Max between 20N and 60N
            amoc_max = np.max(moc_sv[:, idx_20n:idx_60n])
            amoc_series.append(amoc_max)
            
            if i % 12 == 0:
                print(f"Processed Year {year:.1f}: SAT={t_sat:.2f}, AMOC={amoc_max:.2f}")
                
        except Exception as e:
            print(f"Error processing {f}: {e}")
            break

    # --- Plotting Time Series ---
    
    # 1. SAT
    plt.figure(figsize=(10, 5))
    plt.plot(times, t_global_series, label='Global Mean SAT')
    plt.xlabel('Year')
    plt.ylabel('Temperature (K)')
    plt.title('Global Mean Surface Air Temperature Evolution')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/timeseries_sat.png')
    plt.close()
    
    # 2. SST
    plt.figure(figsize=(10, 5))
    plt.plot(times, sst_global_series, color='orange', label='Global Mean SST')
    plt.xlabel('Year')
    plt.ylabel('Temperature (K)')
    plt.title('Global Mean Sea Surface Temperature Evolution')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/timeseries_sst.png')
    plt.close()

    # 3. AMOC
    plt.figure(figsize=(10, 5))
    plt.plot(times, amoc_series, color='green', label='AMOC Index')
    plt.xlabel('Year')
    # Use actual max/min for y-limits or let auto
    plt.ylabel('Transport (Sv)')
    plt.title('AMOC Strength (Max 20N-60N)')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/timeseries_amoc.png')
    plt.close()

    # --- Spatial Maps (Climatology) ---
    
    print("Generating Climatology Maps (Last 5 Years)...")
    
    # Load last 60 files (5 years) for Climatology
    climatology_files = files[-60:]
    if len(climatology_files) < 12:
        print("Warning: Less than 1 year of data for climatology. Using available.")
        climatology_files = files
        
    # Compute Mean
    # We can use xr.open_mfdataset but might be slow/heavy? 
    # Let's straightforwardly sum and divide to avoid memory issues if any, 
    # or trust open_mfdataset for 60 files (should be fine for T31).
    
    # Compute Mean Manual Loop (No Dask)
    print(f"Computing climatology from {len(climatology_files)} files manually...")
    
    # Initialize with first file
    ds_sum = xr.open_dataset(climatology_files[0], decode_times=False)
    # We only need the vars we plot
    vars_needed = ['ocean_temp', 'ocean_salt', 'atmos_temp']
    
    # Create valid sum dataset structure
    ds_mean = ds_sum[vars_needed].copy(deep=True)
    
    # Iterate and Sum
    for f in climatology_files[1:]:
        ds_curr = xr.open_dataset(f, decode_times=False)
        for v in vars_needed:
            ds_mean[v] = ds_mean[v] + ds_curr[v]
            
    # Divide
    count = len(climatology_files)
    for v in vars_needed:
        ds_mean[v] = ds_mean[v] / count
        
    print("Climatology computation complete.")

    ds_first = xr.open_dataset(files[0], decode_times=False)
    
    # Mask for Ocean
    print("Loading Bathymetry Mask...")
    mask_land = load_bathymetry_mask_standalone(nlat=ny, nlon=nx)
    
    # Variables to plot
    vars_to_plot = [
        {'name': 'SST', 'var': 'ocean_temp', 'isel_args': {'z': 0}, 'cmap': 'RdYlBu_r', 'vmin': 270, 'vmax': 305, 'units': 'K'},
        {'name': 'SSS', 'var': 'ocean_salt', 'isel_args': {'z': 0}, 'cmap': 'viridis', 'vmin': 33, 'vmax': 37, 'units': 'psu'},
        {'name': 'SAT', 'var': 'atmos_temp', 'isel_args': {}, 'cmap': 'coolwarm', 'vmin': 230, 'vmax': 310, 'units': 'K'},
    ]
    
    for v in vars_to_plot:
        data_first = ds_first[v['var']].isel(**v['isel_args']).values
        data_clim = ds_mean[v['var']].isel(**v['isel_args']).values
        
        if 'ocean' in v['var']:
             # Strict Masking: Set Land to NaN
             data_first = np.where(mask_land, data_first, np.nan)
             data_clim = np.where(mask_land, data_clim, np.nan)
        
        # 1. Plot Climatology
        current_cmap = matplotlib.colormaps[v['cmap']]
        current_cmap.set_bad(color='gray')
        
        plt.figure(figsize=(10, 6))
        plt.imshow(data_clim, origin='lower', cmap=current_cmap, vmin=v['vmin'], vmax=v['vmax'])
        plt.colorbar(label=f"{v['name']} ({v['units']})")
        plt.title(f"Climatology {v['name']} (Years {times[-len(climatology_files)]:.1f}-{times[-1]:.1f})")
        plt.savefig(f"{output_dir}/map_{v['name']}_climatology.png")
        plt.close()
        
        # 2. Plot Drift (Climatology - Initial)
        diff = data_clim - data_first
        if np.all(np.isnan(diff)):
            limit = 1.0
        else:
            limit = np.nanmax(np.abs(diff))
            if limit > 10.0 and 'ocean' in v['var']: 
                 limit = 5.0
        
        plt.figure(figsize=(10, 6))
        plt.imshow(diff, origin='lower', cmap='seismic', vmin=-limit, vmax=limit)
        plt.colorbar(label=f"Delta {v['name']} ({v['units']})")
        plt.title(f"Drift: {v['name']} (Climatology - Initial)")
        plt.savefig(f"{output_dir}/map_{v['name']}_drift.png")
        plt.close()

    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    analyze_prod_run()
