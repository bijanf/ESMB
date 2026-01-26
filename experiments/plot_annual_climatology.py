
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm import data
import matplotlib.pyplot as plt
import numpy as np

def plot_annual_climatology():
    output_dir = Path("outputs/production_control")
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    
    if len(mean_files) < 12:
        print("Not enough data for annual climatology yet.")
        return

    # Take the last 12 months
    last_12 = mean_files[-12:]
    print(f"Averaging months {last_12[0].name} to {last_12[-1].name}...")
    
    # Accumulate fields
    sum_sst = None
    
    for f in last_12:
        state = model_io.load_state_from_netcdf(f)
        if sum_sst is None:
            sum_sst = np.array(state.fluxes.sst)
        else:
            sum_sst += np.array(state.fluxes.sst)
            
    annual_sst = sum_sst / 12.0
    
    # Plot
    mask = data.load_bathymetry_mask(nz=15)
    sst_masked = np.where(mask, annual_sst, np.nan)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(sst_masked, origin='lower', cmap='RdYlBu_r', vmin=270, vmax=315)
    plt.colorbar(label='SST (K)')
    plt.title(f'Annual Mean SST Climatology (Year {len(mean_files)//12})')
    plt.savefig("annual_climatology_sst.png")
    print("Saved annual_climatology_sst.png")

if __name__ == "__main__":
    plot_annual_climatology()
