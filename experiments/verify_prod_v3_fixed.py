
import xarray as xr
import numpy as np
import glob
import os

def verify_fixed_run():
    output_dir = "outputs/control_run_prod_v3_fixed"
    files = sorted(glob.glob(os.path.join(output_dir, "mean_*.nc")))
    
    if not files:
        print("No files found.")
        return

    # Check the latest file
    last_file = files[-1]
    print(f"Checking latest file: {last_file}")
    
    try:
        ds = xr.open_dataset(last_file, decode_times=False)
        
        # Check SSS
        if 'ocean_salt' in ds:
            # Handle dimensions
            salt = ds.ocean_salt
            if salt.ndim == 4:
                sss = salt.isel(time=-1, z=0).values
            elif salt.ndim == 3:
                sss = salt.isel(z=0).values
            else:
                sss = salt.values
            
            # Laplacian for noise
            sss_roll_xm = np.roll(sss, 1, axis=1)
            sss_roll_xp = np.roll(sss, -1, axis=1)
            sss_roll_ym = np.roll(sss, 1, axis=0)
            sss_roll_yp = np.roll(sss, -1, axis=0)
            
            laplacian = np.abs(sss_roll_xp + sss_roll_xm + sss_roll_yp + sss_roll_ym - 4*sss)
            
            valid_stats = sss > 10.0
            
            print(f"Global Mean Temp: {ds.atmos_temp.mean().values:.2f} K")
            print(f"SSS Min: {np.min(sss):.2f} PSU")
            print(f"SSS Max: {np.max(sss):.2f} PSU")
            
            if np.sum(valid_stats) > 0:
                mean_noise = np.mean(laplacian[valid_stats])
                max_noise = np.max(laplacian[valid_stats])
                print(f"SSS Mean Noise (Laplacian): {mean_noise:.4f} PSU")
                print(f"SSS Max Noise (Laplacian):  {max_noise:.4f} PSU")
                
                if mean_noise < 2.0:
                    print("STATUS: STABLE (Noise < 2.0 PSU)")
                else:
                    print("STATUS: WARNING (Noise High)")
            else:
                print("No valid ocean points.")
                
        else:
            print("ocean_salt variable not found.")
            
    except Exception as e:
        print(f"Error opening {last_file}: {e}")

if __name__ == "__main__":
    verify_fixed_run()
