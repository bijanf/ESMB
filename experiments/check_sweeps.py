
import xarray as xr
import numpy as np
import glob
import os

def check_sweeps():
    dirs = base_dirs = glob.glob("outputs/control_run_sweep_*")
    dirs.sort()
    
    print(f"Checking {len(dirs)} directories...")
    print(f"{'Directory':<40} | {'Max V (m/s)':<15} | {'AMOC Proxy (Sv)':<15}")
    print("-" * 80)
    
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, "mean_*.nc")))
        if not files:
            continue
            
        last_file = files[-1]
        try:
            with xr.open_dataset(last_file, decode_times=False, engine='netcdf4') as ds:
                if 'ocean_v' in ds:
                    v = ds['ocean_v'].values
                    max_v = np.max(np.abs(v))
                    
                    # Rough AMOC Proxy (Sum of V * dx * dz at mid-lat)
                    # Very rough, just to see if it's non-zero
                    amoc_proxy = np.sum(v) * 1e-6 # Meaningless units but non-zero check
                    
                    print(f"{os.path.basename(d):<40} | {max_v:.4f}          | {amoc_proxy:.2f}")
                else:
                    print(f"{os.path.basename(d):<40} | No ocean_v")
        except Exception as e:
            print(f"{os.path.basename(d):<40} | Error: {e}")

if __name__ == "__main__":
    check_sweeps()
