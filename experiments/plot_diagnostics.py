import sys
import os
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import glob

# Try to import Chronos mask
try:
    from chronos_esm import data
    from chronos_esm import config
    HAS_MASK_LOADER = True
except ImportError:
    print("Could not import chronos_esm.data. Assuming no mask.")
    HAS_MASK_LOADER = False

MASK_CACHE = None

def get_mask(ny, nx):
    global MASK_CACHE
    if MASK_CACHE is not None:
        return MASK_CACHE
        
    if HAS_MASK_LOADER:
        try:
            # Mask is (ny, nx) based on config.OCEAN_GRID
            # 1 = Ocean, 0 = Land
            mask = data.load_bathymetry_mask()
            # convert to numpy if jax
            if hasattr(mask, "device_buffer"):
                mask = np.array(mask)
            MASK_CACHE = mask
            return mask
        except Exception as e:
            print(f"Failed to load mask: {e}")
            return None
    return None

def process_file(file_path):
    ds = xr.open_dataset(file_path, decode_times=False)
    
    # helper to find dimension
    def get_dim_name(possible_names, ds_dims):
        for n in possible_names:
            if n in ds_dims:
                return n
        return None

    # ATMOS Processing
    y_dim_atm = get_dim_name(['y_atm', 'y', 'dim_1'], ds.dims)
    if y_dim_atm:
        ny = ds.dims[y_dim_atm]
        lat_atm = np.linspace(-90, 90, ny)
        weights_atm = np.cos(np.deg2rad(lat_atm))
        w_atm = xr.DataArray(weights_atm, dims=y_dim_atm)
    else:
        # Fallback (unlikely)
        w_atm = None
    
    # Mask
    # Assume standard grid for now if mask is needed
    if y_dim_atm:
         nx = ds.dims.get('x_atm', ds.dims.get('x', ds.dims.get('dim_2', 96))) # Fallback
         mask = get_mask(ny, nx)
    else:
         mask = None

    t_atm = ds['atmos_temp']
    if w_atm is not None and t_atm.ndim >= 2:
        # Check if t_atm has the y_dim
        if y_dim_atm in t_atm.dims:
             # Broadcast mask?
             # mask is (ny, nx)
             # weights is (ny,) -> (ny, 1) * mask
             
             w_2d = w_atm
             if mask is not None:
                 # Ensure shapes match
                 # w_atm is DataArray with name 'weights' and dim y_dim_atm
                 # we need 2d weights: cos(lat) * mask
                 
                 # Create dimensions for mask
                 # mask is numpy array
                 # t_atm dims are (y_atm, x_atm) usually
                 
                 # Find x dim
                 x_dim = get_dim_name(['x_atm', 'x', 'dim_2'], t_atm.dims)
                 if x_dim:
                     mask_da = xr.DataArray(mask, dims=[y_dim_atm, x_dim])
                     w_2d = w_atm * mask_da
                 else:
                     # fallback
                     mask_da = xr.DataArray(mask, dims=t_atm.dims)
                     w_2d = w_atm * mask_da # might broadcast
             
             # Computed Weighted Mean
             # Sum of weights:
             sum_weights = w_2d.sum()
             weighted_sum = (t_atm * w_2d).sum()
             t_global = (weighted_sum / sum_weights).item()
        else:
            t_global = t_atm.mean().item()
    else:
        t_global = t_atm.mean().item()
        
    # SST
    # distinct variable or proxy
    # In this run, fluxes_sst is missing. Use ocean_temp[0].
    
    # OCEAN Processing
    y_dim_ocn = get_dim_name(['y_ocn', 'y', 'dim_1', 'dim_2'], ds.dims)
    z_dim_ocn = get_dim_name(['z', 'dim_0'], ds.dims)
    
    if y_dim_ocn:
        ny = ds.dims[y_dim_ocn]
        lat_ocn = np.linspace(-90, 90, ny)
        weights_ocn = np.cos(np.deg2rad(lat_ocn))
        w_ocn = xr.DataArray(weights_ocn, dims=y_dim_ocn)
    else:
        w_ocn = None
        
    o_temp = ds['ocean_temp']
    
    # SST Proxy (Surface)
    if z_dim_ocn in o_temp.dims:
         sst_proxy = o_temp.isel({z_dim_ocn: 0})
    else:
         sst_proxy = o_temp
         
    if w_ocn is not None and y_dim_ocn in sst_proxy.dims:
        sst = sst_proxy.weighted(w_ocn).mean().item()
    else:
        sst = sst_proxy.mean().item()
    
    # Deep Ocean (Bottom)
    if z_dim_ocn in o_temp.dims:
        o_deep_field = o_temp.isel({z_dim_ocn: -1})
    else:
        o_deep_field = o_temp
        
    if w_ocn is not None and y_dim_ocn in o_deep_field.dims:
        o_deep = o_deep_field.weighted(w_ocn).mean().item()
        o_vol = o_temp.weighted(w_ocn).mean().item()
    else:
        o_deep = o_deep_field.mean().item()
        o_vol = o_temp.mean().item()

    return {
        'atmos_temp': t_global,
        'sst': sst,
        'ocean_deep': o_deep,
        'ocean_vol': o_vol
    }

def run_analysis(data_dir="outputs/control_run"):
    files = sorted(glob.glob(os.path.join(data_dir, "mean_*.nc")))
    if not files:
        print("No files found!")
        return

    results = {'atmos_temp': [], 'sst': [], 'ocean_deep': [], 'ocean_vol': [], 'time': []}
    
    for i, f in enumerate(files):
        print(f"Processing {f}...")
        try:
            res = process_file(f)
            for k, v in res.items():
                results[k].append(v)
            results['time'].append(i+1)
        except Exception as e:
            print(f"Failed to process {f}: {e}")
            continue
            
    # Filter outliers (Start from month 3)
    # start_idx = 2
    # for k in results:
    #     results[k] = results[k][start_idx:]
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    t_arr = results['time']
    
    axes[0].plot(t_arr, results['atmos_temp'], label='Atmos Temp (Global)', linewidth=2)
    axes[0].plot(t_arr, results['sst'], label='SST (Global)', linestyle='--')
    axes[0].set_ylabel('Temperature (K)')
    axes[0].set_title('Surface Temperatures')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(t_arr, results['ocean_deep'], label='Deep Ocean Temp', color='purple')
    axes[1].set_ylabel('Temperature (K)')
    axes[1].set_title('Deep Ocean Stability')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(t_arr, results['ocean_vol'], label='Ocean Volume Mean', color='green')
    axes[2].set_ylabel('Temperature (K)')
    axes[2].set_title('Ocean Heat Content / Volume Mean')
    axes[2].set_xlabel('Time (Months)')
    axes[2].legend()
    axes[2].grid(True)
    
    output_file = os.path.join(data_dir, "diagnostics.png")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    run_analysis()
