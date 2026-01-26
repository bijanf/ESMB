
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def check_noise():
    # Load the last mean file
    ds = xr.open_dataset('outputs/control_run_verify_noise/mean_0001.nc', decode_times=False)
    
    # Get SST (ocean_temp at z=0)
    sst = ds.ocean_temp.isel(z=0).values
    
    # Simple Laplacian filter to detect grid scale noise
    # L(T) = T(i+1) + T(i-1) + T(j+1) + T(j-1) - 4*T
    
    sst_roll_xm = np.roll(sst, 1, axis=1)
    sst_roll_xp = np.roll(sst, -1, axis=1)
    sst_roll_ym = np.roll(sst, 1, axis=0)
    sst_roll_yp = np.roll(sst, -1, axis=0)
    
    laplacian = np.abs(sst_roll_xp + sst_roll_xm + sst_roll_yp + sst_roll_ym - 4*sst)
    
    # Ignore land (where sst is 283.15 or similar constant/masked)
    # Assuming mask can be inferred or loaded. 
    # Let's just look at max values.
    
    print(f"Max Laplacian (Noise Metric): {np.max(laplacian):.4f} K")
    print(f"Mean Laplacian: {np.mean(laplacian):.4f} K")
    print(f"99th Percentile Laplacian: {np.percentile(laplacian, 99):.4f} K")
    
    # Identify "dots" - points significantly different from neighbors
    threshold = 2.0 # K difference from neighbor avg
    dots = np.where(laplacian > threshold)
    print(f"Number of noisy points (> {threshold}K delta): {len(dots[0])}")

if __name__ == "__main__":
    check_noise()
