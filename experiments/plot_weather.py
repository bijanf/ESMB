
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

import glob

def plot_weather():
    # Find latest file
    files = sorted(glob.glob("outputs/control_run_v3_partial/mean_*.nc"))
    if not files:
        print("No files found!")
        return
    
    last_file = files[-1]
    print(f"Using file: {last_file}")
    
    try:
        ds = xr.open_dataset(last_file, decode_times=False)
    except Exception as e:
        print(f"Error opening file: {e}")
        return
        
    print("Keys:", list(ds.keys()))
    
    # Plot Vorticity (Atmos)
    if 'atmos_vorticity' in ds:
        vort = ds['atmos_vorticity'].values
        # Level? 
        if vort.ndim == 3: # (z, y, x)? No atmos is usually 2D or layers?
            # Atmos dynamics 1 layer or multiple?
            # primitive equations usually multi-layer.
            # But the output in io.py says "y_atm", "x_atm" for atmos variables?
            # Let's check shape.
            pass
        
        plt.figure(figsize=(10, 6))
        plt.imshow(vort, origin='lower', cmap='RdBu_r')
        plt.colorbar(label='Vorticity (s-1)')
        plt.title('Monthly Mean Vorticity (Month 462)')
        plt.savefig('vorticity_map.png')
        print("Saved vorticity_map.png")
        
    # Plot Temp Deviation from Zonal Mean (Eddies)
    if 'atmos_temp' in ds:
        temp = ds['atmos_temp'].values
        z_idx = 0 # Surface? Or mean? 
        # temp shape: (y_atm, x_atm) or (z, y, x)?
        # io.py saves (y_atm, x_atm). It's 2D.
        
        zonal_mean = np.mean(temp, axis=-1, keepdims=True)
        eddies = temp - zonal_mean
        
        plt.figure(figsize=(10, 6))
        plt.imshow(eddies, origin='lower', cmap='RdBu_r', vmin=-10, vmax=10)
        plt.colorbar(label='Temp Deviation from Zonal Mean (K)')
        plt.title('Temperature Eddies (Month 462)')
        plt.savefig('eddies_map.png')
        print("Saved eddies_map.png")

if __name__ == "__main__":
    plot_weather()
