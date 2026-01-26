
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from chronos_esm import data

def plot_verify_map():
    ds = xr.open_dataset('outputs/control_run_verify_noise/mean_0001.nc', decode_times=False)
    sst = ds.ocean_temp.isel(z=0).values
    
    mask_land = np.array(data.load_bathymetry_mask())
    sst_masked = np.ma.masked_where(~mask_land, sst)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(sst_masked, origin='lower', cmap='RdYlBu_r', vmin=270, vmax=305)
    plt.colorbar(label='SST (K)')
    plt.title('SST with Horizontal Diffusion (kappa_h=100)')
    plt.savefig('verify_noise_sst.png')
    print("Saved verify_noise_sst.png")

if __name__ == "__main__":
    plot_verify_map()
