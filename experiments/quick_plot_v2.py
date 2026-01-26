
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add path for mask loading if needed
sys.path.append(os.getcwd())
from experiments.inspect_climatology import load_bathymetry_mask_standalone

def plot_v2_snapshot():
    print("Plotting v2 snapshot...")
    path = "outputs/control_run_prod_v2/mean_0012.nc"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return

    ds = xr.open_dataset(path, decode_times=False)
    sst = ds.ocean_temp.isel(z=0).values
    
    # Mask
    ny, nx = sst.shape
    mask = load_bathymetry_mask_standalone(nlat=ny, nlon=nx)
    sst_masked = np.where(mask, sst, np.nan)
    
    plt.figure(figsize=(10, 6))
    # Use a tight range to trigger contrast on noise
    plt.imshow(sst_masked, origin='lower', cmap='RdYlBu_r', vmin=270, vmax=305)
    plt.colorbar(label='SST (K)')
    plt.title("Prod v2 (Year 1) - SST Smoothness Check")
    plt.savefig("map_v2_check.png")
    print("Saved map_v2_check.png")

if __name__ == "__main__":
    plot_v2_snapshot()
