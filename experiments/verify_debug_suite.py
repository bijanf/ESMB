
import xarray as xr
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def verify_debug_run(run_name, label):
    output_dir = f"outputs/control_run_{run_name}"
    files = sorted(glob.glob(os.path.join(output_dir, 'mean_*.nc')))
    
    if not files:
        print(f"[{label}] No files found in {output_dir}")
        return None
        
    last_file = files[-1]
    print(f"[{label}] Checking {last_file}...")
    
    try:
        ds = xr.open_dataset(last_file, decode_times=False)
        
        # SSS Noise Check
        if 'ocean_salt' in ds:
            # Handle dimensions (time, z, y, x) or varying
            salt = ds.ocean_salt
            if salt.ndim == 4: # time, z, y, x
                sss = salt.isel(time=-1, z=0).values
            elif salt.ndim == 3: # z, y, x
                sss = salt.isel(z=0).values
            else:
                sss = salt.values
                
            # Laplacian
            sss_roll_xm = np.roll(sss, 1, axis=1)
            sss_roll_xp = np.roll(sss, -1, axis=1)
            sss_roll_ym = np.roll(sss, 1, axis=0)
            sss_roll_yp = np.roll(sss, -1, axis=0)
            
            laplacian = np.abs(sss_roll_xp + sss_roll_xm + sss_roll_yp + sss_roll_ym - 4*sss)
            
            # Mask
            valid_mask = sss > 10.0
            
            if np.sum(valid_mask) > 0:
                mean_noise = np.mean(laplacian[valid_mask])
                max_noise = np.max(laplacian[valid_mask])
                p99_noise = np.percentile(laplacian[valid_mask], 99)
                
                print(f"[{label}] SSS Mean Noise: {mean_noise:.4f} PSU")
                print(f"[{label}] SSS Max Noise:  {max_noise:.4f} PSU")
                print(f"[{label}] SSS P99 Noise:  {p99_noise:.4f} PSU")
                
                return {
                    'run': label,
                    'mean_noise': mean_noise,
                    'max_noise': max_noise
                }
            else:
                print(f"[{label}] No valid ocean points found.")
        else:
            print(f"[{label}] ocean_salt not found.")
            
    except Exception as e:
        print(f"[{label}] Error: {e}")
        return None

def main():
    results = []
    
    r1 = verify_debug_run("debug_base", "Baseline (kH=1000)")
    if r1: results.append(r1)
    
    r2 = verify_debug_run("debug_diff3k", "Moderate (kH=3000)")
    if r2: results.append(r2)
    
    r3 = verify_debug_run("debug_diff5k", "High (kH=5000)")
    if r3: results.append(r3)

    r4 = verify_debug_run("debug_kh15k", "High+ (kH=15000)")
    if r4: results.append(r4)

    r5 = verify_debug_run("debug_kh20k", "High++ (kH=20000)")
    if r5: results.append(r5)
    
    print("\n--- Summary ---")
    print(f"{'Run':<20} | {'Mean Noise':<12} | {'Max Noise':<12}")
    print("-" * 50)
    for r in results:
        print(f"{r['run']:<20} | {r['mean_noise']:.4f}       | {r['max_noise']:.4f}")

if __name__ == "__main__":
    main()
