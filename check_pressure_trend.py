
import xarray as xr
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

OUTPUT_DIR = "outputs/production_control"

def check_pressure():
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "mean_*.nc")))
    print(f"Found {len(files)} files.")
    
    times = []
    mean_ps = []
    
    # Check every 10th file to save time
    for i, f in enumerate(files[::10]):
        ds = xr.open_dataset(f, decode_times=False)
        if i == 0:
             print("Keys:", list(ds.data_vars.keys()))
             
        if 'atmos_ln_ps' in ds:
            ln_ps = ds['atmos_ln_ps'].values
            ps = np.exp(ln_ps)
            ps_mean = np.mean(ps)
            
            basename = os.path.basename(f)
            month = int(basename.split('_')[1].split('.')[0])
            
            times.append(month)
            mean_ps.append(ps_mean)
            print(f"Month {month}: Mean P = {ps_mean:.2f} Pa")
        else:
            print(f"atmos_ln_ps not found based on check. Keys: {list(ds.data_vars.keys())}")
        ds.close()
        ds.close()

    plt.figure()
    plt.plot(times, mean_ps)
    plt.xlabel('Month')
    plt.ylabel('Mean Surface Pressure (Pa)')
    plt.title('Global Mean Pressure Trend')
    plt.grid(True)
    plt.savefig('pressure_trend.png')
    print("Saved pressure_trend.png")

if __name__ == "__main__":
    check_pressure()
