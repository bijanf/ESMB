
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm import data
import matplotlib.pyplot as plt
import numpy as np

def plot_sst_anomaly():
    output_dir = Path("outputs/production_control")
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    if not mean_files: return

    # Use last 12 months for annual mean
    last_12 = mean_files[-12:]
    print(f"Calculating anomaly for {last_12[-1].name} (Annual Mean)...")
    
    sum_sst = None
    for f in last_12:
        state = model_io.load_state_from_netcdf(f)
        if sum_sst is None: sum_sst = np.array(state.fluxes.sst)
        else: sum_sst += np.array(state.fluxes.sst)
    sst = sum_sst / 12.0
    
    mask = data.load_bathymetry_mask(nz=15)
    
    # Calculate Zonal Mean
    # Mask land with NaN
    sst_masked = np.where(mask, sst, np.nan)
    zonal_mean = np.nanmean(sst_masked, axis=1, keepdims=True)
    
    # Anomaly
    anomaly = sst_masked - zonal_mean
    
    plt.figure(figsize=(12, 6))
    # Reduced range to +/- 2 K to show gyre structure
    plt.imshow(anomaly, origin='lower', cmap='RdBu_r', vmin=-2, vmax=2)
    plt.colorbar(label='SST Anomaly (K)')
    plt.contour(anomaly, levels=np.linspace(-2, 2, 11), colors='k', linewidths=0.5, alpha=0.3)
    plt.title('SST Zonal Anomaly (Deviation from Latitude Mean)')
    plt.savefig("sst_zonal_anomaly.png")
    print("Saved sst_zonal_anomaly.png")

if __name__ == "__main__":
    plot_sst_anomaly()
