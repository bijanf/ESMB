import xarray as xr
import numpy as np
import sys

def check_temps(filename):
    print(f"Checking {filename}...")
    try:
        ds = xr.open_dataset(filename, decode_times=False, engine='h5netcdf')
    except Exception as e:
        print(f"h5netcdf failed: {e}")
        # Fallback for netcdf4
        ds = xr.open_dataset(filename, decode_times=False)

    print("Keys:", list(ds.keys()))
    
    # Approx mask if not in file
    # But land_temp and ocean_temp should be there
    
    if 'land_temp' in ds:
        t_land = ds['land_temp'].values
        # Mask where 0 (if masked) or check values
        print(f"Land Temp Mean: {np.mean(t_land):.2f} K")
        print(f"Land Temp Max: {np.max(t_land):.2f} K")
        print(f"Land Temp Min: {np.min(t_land):.2f} K")
        
    if 'ocean_temp' in ds:
        t_ocean = ds['ocean_temp'].values
        # Layer 0
        if t_ocean.ndim == 4: t_ocean = t_ocean[0,0] # Time, Z, Y, X
        elif t_ocean.ndim == 3: t_ocean = t_ocean[0] # Z, Y, X or Time, Y, X? Check code. 
        # Usually (Time, Z, Y, X) for saved state
        
        print(f"Ocean Temp Mean (SST): {np.mean(t_ocean):.2f} K (approx)")
        print(f"Ocean Temp Max (SST): {np.max(t_ocean):.2f} K")
        
    if 'atmos_temp' in ds:
        t_atm = ds['atmos_temp'].values
        print(f"Atmos Temp Mean: {np.mean(t_atm):.2f} K")
        
    if 'atmos_u' in ds:
        u = ds['atmos_u'].values
        v = ds['atmos_v'].values
        ws = np.sqrt(u**2 + v**2)
        print(f"Wind Speed Mean: {np.mean(ws):.2f} m/s")
        print(f"Wind Speed Max: {np.max(ws):.2f} m/s")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_temps(sys.argv[1])
    else:
        check_temps('outputs/production_control/mean_0005.nc')
