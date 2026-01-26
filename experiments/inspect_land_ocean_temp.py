
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm import data
import matplotlib.pyplot as plt
import numpy as np

def inspect_temps():
    output_dir = Path("outputs/production_control")
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    if not mean_files: return

    latest_file = mean_files[-1]
    print(f"Analyzing {latest_file}...")
    state = model_io.load_state_from_netcdf(latest_file)
    sst = state.fluxes.sst
    
    # Load Mask
    mask = data.load_bathymetry_mask(nz=15) # True=Ocean
    
    # Convert to numpy for easier handling
    sst_np = np.array(sst)
    mask_np = np.array(mask)
    
    ocean_temps = sst_np[mask_np]
    land_temps = sst_np[~mask_np]
    
    print(f"Ocean Temp: Min={ocean_temps.min():.2f}, Max={ocean_temps.max():.2f}, Mean={ocean_temps.mean():.2f}, Std={ocean_temps.std():.2f}")
    print(f"Land Temp:  Min={land_temps.min():.2f}, Max={land_temps.max():.2f}, Mean={land_temps.mean():.2f}, Std={land_temps.std():.2f}")
    
    # Plot Ocean Only
    sst_ocean = np.where(mask_np, sst_np, np.nan)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(sst_ocean, origin='lower', cmap='RdYlBu_r')
    plt.colorbar(label='Ocean Temp (K)')
    plt.title(f'Ocean Temperature Only - {latest_file.name}')
    plt.savefig("debug_sst_masked.png")
    
    # Plot Land Only
    sst_land = np.where(~mask_np, sst_np, np.nan)
    plt.figure(figsize=(10, 6))
    plt.imshow(sst_land, origin='lower', cmap='RdYlBu_r')
    plt.colorbar(label='Land Temp (K)')
    plt.title(f'Land Temperature Only - {latest_file.name}')
    plt.savefig("debug_land_only.png")

if __name__ == "__main__":
    inspect_temps()
