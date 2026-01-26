
import xarray as xr
import numpy as np
import sys

def check_active_ocean(filename):
    print(f"Checking Ocean Activity in {filename}...")
    ds = xr.open_dataset(filename, decode_times=False)
    
    print("Available variables:", list(ds.keys()))
    
    # Identify Velocity Variables
    if 'ocean_v' in ds:
        v = ds.ocean_v.values
        u = ds.ocean_u.values
    elif 'v' in ds:
        v = ds.v.values
        u = ds.u.values
    else:
        print("Error: No velocity fields found.")
        return

    # Statistics
    v_max = np.nanmax(v)
    v_min = np.nanmin(v)
    v_rms = np.sqrt(np.nanmean(v**2))
    
    u_max = np.nanmax(u)
    
    print(f"Meridional Velocity (v): Max={v_max:.4f} m/s, Min={v_min:.4f} m/s, RMS={v_rms:.5f}")
    print(f"Zonal Velocity (u): Max={u_max:.4f} m/s")
    
    # Status
    if v_max > 0.05: # 5 cm/s
        print("Status: ACTIVE FLOW (Likely Healthy)")
    elif v_max > 0.01:
        print("Status: WEAK FLOW (Struggling)")
    else:
        print("Status: DEAD OCEAN (Stagnant)")

if __name__ == "__main__":
    check_active_ocean(sys.argv[1])
