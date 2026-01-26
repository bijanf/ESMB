
import xarray as xr
import numpy as np
import os
import glob
import sys
from pathlib import Path

# Add parent path to allow importing chronos_esm if needed (though we just use xarray here)
sys.path.insert(0, str(Path(__file__).parent.parent))

def compute_roughness(arr):
    """
    Compute a roughness metric based on the second difference (Laplacian-like)
    in the latitudinal direction.
    High values indicate zigzag/sawtooth patterns.
    """
    # arr shape: (nlat,)
    # Second difference: T(i+1) - 2T(i) + T(i-1)
    d2 = np.diff(arr, n=2)
    return np.std(d2)

def check_noise(run_name, start_month=1, end_month=5):
    output_dir = f"outputs/{run_name}"
    
    files = []
    # Try to find all available files if end_month is not strictly enforced
    # or just look for the specific range
    for m in range(start_month, end_month + 1):
        f = os.path.join(output_dir, f"mean_{m:04d}.nc")
        if os.path.exists(f):
            files.append(f)
    
    if not files:
        print(f"No files found in {output_dir} for range {start_month}-{end_month}")
        return

    print(f"Analyzing {len(files)} files from {run_name}...")
    
    print(f"{'Month':<6} | {'Var':<6} | {'Global Mean':<12} | {'Zonal Roughness':<15} | {'Status':<10}")
    print("-" * 65)

    for f in files:
        month = int(os.path.basename(f).split('_')[1].split('.')[0])
        ds = xr.open_dataset(f, decode_times=False)
        
        # Check Atmosphere Temp
        if 'atmos_temp' in ds:
            # Assuming (lev, lat, lon) or (lat, lon)
            temp = ds.atmos_temp.values
            if temp.ndim == 3: temp = temp[0] # Surface
            
            # Zonal Mean
            zonal_mean = np.mean(temp, axis=1)
            gmean = np.mean(temp)
            
            # Roughness
            roughness = compute_roughness(zonal_mean)
            
            status = "OK"
            if roughness > 1.0: status = "NOISY"
            if roughness > 5.0: status = "UNSTABLE"
            
            print(f"{month:<6} | {'SAT':<6} | {gmean:<12.2f} | {roughness:<15.4f} | {status:<10}")

        # Check Ocean Temp (SST)
        if 'ocean_temp' in ds:
            # Ocean usually has mask 0, need to be careful with mean
            # Assuming 3D (z, y, x)
            temp = ds.ocean_temp.values
            if temp.ndim == 3: temp = temp[0] # Surface
            
            # Simple mean including zeros? No, mask 0s usually.
            # But the model output might have zeros or NaNs where land is.
            # Let's filter > 0 (Kelvin)
            mask = temp > 100.0 
            
            # Zonal mean of masked data
            zonal_mean = []
            for j in range(temp.shape[0]):
                row = temp[j, :]
                valid = row[row > 100.0]
                if len(valid) > 0:
                    zonal_mean.append(np.mean(valid))
                else:
                    # If no ocean at this lat, use neighbor or NaN
                    # For roughness calc, this breaks it.
                    # Just append previous or 273.15
                    zonal_mean.append(273.15) 
            
            zonal_mean = np.array(zonal_mean)
            gmean = np.mean(temp[mask])
            
            roughness = compute_roughness(zonal_mean)
            
            # Ocean is usually smoother
            status = "OK"
            if roughness > 0.5: status = "NOISY"
            
            print(f"{month:<6} | {'SST':<6} | {gmean:<12.2f} | {roughness:<15.4f} | {status:<10}")
        
        ds.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="production_control")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=12)
    args = parser.parse_args()
    
    check_noise(args.run_name, args.start, args.end)
