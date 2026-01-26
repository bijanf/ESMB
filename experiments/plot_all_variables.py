
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm import data
import matplotlib.pyplot as plt
import numpy as np

def plot_all():
    output_dir = Path("outputs/production_control")
    results_dir = Path("analysis_results/prod_v6")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    if not mean_files: return

    # Plot Latest
    f = mean_files[-1]
    print(f"Plotting {f.name}...")
    state = model_io.load_state_from_netcdf(f)
    mask = data.load_bathymetry_mask(nz=15)
    
    # 1. Sea Ice
    ice = np.array(state.ice.concentration)
    ice_masked = np.where(mask, ice, np.nan)
    plt.figure(figsize=(10, 5))
    plt.imshow(ice_masked, origin='lower', cmap='Blues_r', vmin=0, vmax=1)
    plt.colorbar(label='Concentration')
    plt.title(f'Sea Ice Concentration ({f.name})')
    plt.savefig(results_dir / "ice_map.png")
    
    # 2. Atmos Temp
    atm_t = np.array(state.atmos.temp)
    plt.figure(figsize=(10, 5))
    plt.imshow(atm_t, origin='lower', cmap='plasma')
    plt.colorbar(label='T (K)')
    plt.title(f'Atmospheric Temperature ({f.name})')
    plt.savefig(results_dir / "atmos_temp_map.png")
    
    # 3. Ocean Salinity
    sss = np.array(state.ocean.salt[0]) # Surface
    sss_masked = np.where(mask, sss, np.nan)
    plt.figure(figsize=(10, 5))
    plt.imshow(sss_masked, origin='lower', cmap='viridis', vmin=33, vmax=37)
    plt.colorbar(label='Salinity (psu)')
    plt.title(f'Sea Surface Salinity ({f.name})')
    plt.savefig(results_dir / "sss_map.png")
    
    # 4. Precipitation
    precip = np.array(state.fluxes.precip) * 86400 # mm/day
    plt.figure(figsize=(10, 5))
    plt.imshow(precip, origin='lower', cmap='YlGnBu', vmin=0, vmax=10)
    plt.colorbar(label='mm/day')
    plt.title(f'Precipitation ({f.name})')
    plt.savefig(results_dir / "precip_map.png")
    
    # 5. Wind Stress
    tau_x = np.array(state.fluxes.wind_stress_x)
    plt.figure(figsize=(10, 5))
    plt.imshow(tau_x, origin='lower', cmap='RdBu_r', vmin=-0.1, vmax=0.1)
    plt.colorbar(label='N/m2')
    plt.title(f'Zonal Wind Stress ({f.name})')
    plt.savefig(results_dir / "wind_stress_map.png")
    
    print("Done.")

if __name__ == "__main__":
    plot_all()
