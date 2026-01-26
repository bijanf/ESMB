
import xarray as xr
import numpy as np
import glob
import os
from pathlib import Path

def check_prod_v3():
    output_dir = 'outputs/control_run_prod_v3'
    # Find all mean files
    files = sorted(glob.glob(os.path.join(output_dir, 'mean_*.nc')))
    
    if not files:
        print("No output files found!")
        return

    # Check specific early file to see if noise starts early
    target_file = "outputs/control_run_prod_v3/mean_0010.nc"
    if not os.path.exists(target_file):
        print(f"{target_file} not found. checking first available.")
        target_file = files[0] # Fallback
        
    print(f"Checking file: {target_file}")
    try:
        ds = xr.open_dataset(target_file, decode_times=False)
    except Exception as e:
        print(f"Error opening {target_file}: {e}")
        return

    # Check for NaNs in Atmos Temp (Global Stability)
    # atmos_temp is (y_atm, x_atm)
    atmos_temp = ds.atmos_temp.values
    if np.isnan(atmos_temp).any():
        print("CRITICAL: NaNs found in Atmos Temp!")
    else:
        print(f"Stability OK. Global Mean Temp: {np.mean(atmos_temp):.2f} K")

    # Check SSS Noise
    # SSS is ocean_salt at z=0 (surface)
    # ocean_salt shape: (z, y_ocn, x_ocn)
    if 'ocean_salt' in ds:
        sss = ds.ocean_salt.isel(z=0).values
        
        # Laplacian for noise
        # Handle boundaries roughly (this is just a quick check)
        sss_roll_xm = np.roll(sss, 1, axis=1)
        sss_roll_xp = np.roll(sss, -1, axis=1)
        sss_roll_ym = np.roll(sss, 1, axis=0)
        sss_roll_yp = np.roll(sss, -1, axis=0)
        
        laplacian = np.abs(sss_roll_xp + sss_roll_xm + sss_roll_yp + sss_roll_ym - 4*sss)
        
        # Mask land (0 values or specific mask? Usually salt is 0 on land in Veros output often?)
        # Let's assume non-zero values are ocean or use mask if available.
        # Actually usually there is a mask.
        valid_points = sss > 10.0 # Crude mask for ocean (freshwater is rarely that low in open ocean, land is 0)
        
        print(f"SSS Raw Range: {np.min(sss):.2f} to {np.max(sss):.2f} PSU")
        print(f"Number of points > 50 PSU: {np.sum(sss > 50)}")
        
        if np.sum(valid_points) > 0:
            noise_metric = np.mean(laplacian[valid_points])
            max_noise = np.max(laplacian[valid_points])
            print(f"SSS Noise Metric (Mean Laplacian): {noise_metric:.4f} PSU")
            print(f"SSS Max Noise (Max Laplacian): {max_noise:.4f} PSU")
        else:
             print("Could not mask ocean for SSS check.")

    else:
        print("ocean_salt variable not found.")

    # Check AMOC if calculated or inferable (psi)
    if 'ocean_psi' in ds:
        psi = ds.ocean_psi.values
        # AMOC is max of psi usually (or min depending on sign convention)
        print(f"Psi Range: {np.min(psi):.2e} to {np.max(psi):.2e}")

if __name__ == "__main__":
    check_prod_v3()
