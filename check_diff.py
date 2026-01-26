
import xarray as xr
import numpy as np
import sys

def check_diff():
    try:
        ds = xr.open_dataset("outputs/control_run/mean_0051.nc", decode_times=False)
        print("Loaded mean_0051.nc")
        
        # Variables: 'temp' (Atmos), 'sst' (Surface composite)
        # Note: sst in output is usually the composite surface temp seen by atmos
        
        print("Variables:", list(ds.keys()))
        
        if 'atmos_temp' in ds:
            t_atmos = ds['atmos_temp'].values
        elif 'temp' in ds:
             t_atmos = ds['temp'].values
        
        if 'sst' in ds:
             t_surf = ds['sst'].values
        elif 'ocean_temp' in ds:
             # Ocean Temp is (nz, ny, nx) or (time, nz, ny, nx)
             # Mean file has time dim? No, state files usually don't have time dim if saved individually? 
             # Let's check shape.
             ot = ds['ocean_temp'].values
             if len(ot.shape) == 4: # (time, nz, ny, nx)
                 t_surf = ot[:, 0, :, :]
                 t_atmos = t_atmos[:, :, :] # Ensure matching dims if needed
             elif len(ot.shape) == 3: # (nz, ny, nx)
                 t_surf = ot[0, :, :]
             print(f"Using Ocean Temp (Layer 0) as SST proxy. Shape: {t_surf.shape}")
        else:
             print("SST not found!")
             return
            
        diff = t_surf - t_atmos
        
        print(f"Global Mean Diff (Surf - Air): {np.mean(diff):.4f} K")
        print(f"Max Diff: {np.max(diff):.4f} K")
        print(f"Min Diff: {np.min(diff):.4f} K")
        
        # Ocean Only? Need mask.
        # Mask is not in the monthly mean file usually, but we can infer or load it.
        # For now, global stats give a hint. Ocean is 70% of pixels.
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_diff()
