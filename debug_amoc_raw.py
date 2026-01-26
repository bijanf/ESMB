
import xarray as xr
import numpy as np
import os
import glob
import sys

def inspect_file(f):
    if not os.path.exists(f):
        print(f"File {f} not found.")
        return

    print(f"\nInspecting {f}...")
    ds = xr.open_dataset(f, decode_times=False)
    
    # Check Atmsophere
    if 'atmos_temp' in ds:
        t = ds['atmos_temp'].values
        if t.ndim == 3: t = t[0]
        print(f"Atmos T Range: {np.min(t):.2f} to {np.max(t):.2f} K")
        print(f"Atmos T Mean: {np.mean(t):.2f} K")
        
    if 'atmos_u' in ds:
        u = ds['atmos_u'].values
        if u.ndim == 3: u = u[0]
        print(f"Atmos U Range: {np.min(u):.2e} to {np.max(u):.2e} m/s")
        print(f"Atmos U Mean: {np.mean(u):.2e} m/s")
        
    # Check Ocean
    if 'ocean_v' in ds:
        v = ds['ocean_v'].values
        if v.ndim == 4: v = v[0]
        print(f"Ocean V Range: {np.min(v):.2e} to {np.max(v):.2e} m/s")
        
    # Check Ocean Temp (driver of density)
    if 'ocean_temp' in ds:
        ot = ds['ocean_temp'].values
        if ot.ndim == 4: ot = ot[0]
        print(f"Ocean T Range: {np.min(ot):.2f} to {np.max(ot):.2f} K")
        # Check Equator-Pole Gradient (approx)
        ny = ot.shape[1]
        t_equator = np.mean(ot[:, ny//2-2:ny//2+2, :])
        t_pole = np.mean(ot[:, -1, :])
        print(f"Ocean T Equator (Depth Avg): {t_equator:.2f} K")
        print(f"Ocean T Pole (Depth Avg): {t_pole:.2f} K")

def main():
    inspect_file("outputs/production_control/mean_0430.nc")
    inspect_file("outputs/production_control/mean_0444.nc")

if __name__ == "__main__":
    main()
