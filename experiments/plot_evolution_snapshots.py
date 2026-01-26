
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm import data
from chronos_esm.ocean import diagnostics as ocean_diagnostics
import matplotlib.pyplot as plt
import numpy as np

def plot_snapshots():
    output_dir = Path("outputs/production_control")
    results_dir = Path("analysis_results/prod_v6")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    mask = data.load_bathymetry_mask(nz=15)
    
    # Years to plot
    years = [0, 1, 2, 5]
    
    for y in years:
        month = max(1, y * 12)
        if y == 0: month = 1
        
        fname = f"mean_{month:04d}.nc"
        file_path = output_dir / fname
        
        if not file_path.exists():
            print(f"Skipping Year {y} (File {fname} not found)")
            continue
            
        print(f"Plotting Year {y}...")
        state = model_io.load_state_from_netcdf(file_path)
        
        # 1. SST
        sst = np.array(state.fluxes.sst)
        sst_masked = np.where(mask, sst, np.nan)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(sst_masked, origin='lower', cmap='RdYlBu_r', vmin=270, vmax=315)
        plt.colorbar(label='SST (K)')
        plt.title(f'SST - Year {y} (Month {month})')
        plt.savefig(results_dir / f"sst_snapshot_Y{y}.png")
        plt.close()
        
        # 2. Zonal Mean Temperature (Ocean)
        # (nz, ny, nx) -> (nz, ny)
        temp = np.array(state.ocean.temp)
        # Mask land in 3D
        mask_3d = np.broadcast_to(mask, temp.shape)
        temp_masked = np.where(mask_3d, temp, np.nan)
        zonal_mean = np.nanmean(temp_masked, axis=2)
        
        plt.figure(figsize=(8, 6))
        lat = np.linspace(-90, 90, zonal_mean.shape[1])
        depth = np.linspace(0, 5000, zonal_mean.shape[0])
        plt.contourf(lat, depth, zonal_mean, levels=20, cmap='RdYlBu_r')
        plt.colorbar(label='Temp (K)')
        plt.gca().invert_yaxis()
        plt.xlabel('Latitude')
        plt.ylabel('Depth (m)')
        plt.title(f'Ocean Zonal Mean Temp - Year {y}')
        plt.savefig(results_dir / f"zonal_mean_Y{y}.png")
        plt.close()

    print(f"Analysis saved to {results_dir}")

if __name__ == "__main__":
    plot_snapshots()
