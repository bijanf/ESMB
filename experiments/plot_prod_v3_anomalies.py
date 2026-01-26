import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io
import matplotlib.pyplot as plt

def analyze_noise():
    output_dir = Path("outputs/production_control")
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    if not mean_files: return

    latest_file = mean_files[-1]
    print(f"Analyzing noise in {latest_file}...")
    state = model_io.load_state_from_netcdf(latest_file)
    sst = state.fluxes.sst
    
    # 1. Calculate Zonal Mean Anomaly
    zonal_mean = jnp.mean(sst, axis=1, keepdims=True)
    anomaly = sst - zonal_mean
    
    # 2. Calculate Checkerboard Index (High-pass filter)
    # diff between point and average of its 4 neighbors
    smoothed = (jnp.roll(sst, 1, 0) + jnp.roll(sst, -1, 0) + jnp.roll(sst, 1, 1) + jnp.roll(sst, -1, 1)) / 4.0
    noise = sst - smoothed
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    im1 = axes[0].imshow(anomaly, origin='lower', cmap='RdBu_r', vmin=-10, vmax=10)
    axes[0].set_title("SST Anomaly (SST - Zonal Mean)")
    plt.colorbar(im1, ax=axes[0], label='K')
    
    im2 = axes[1].imshow(noise, origin='lower', cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1].set_title("Grid-Scale Noise (High-Pass Filter)")
    plt.colorbar(im2, ax=axes[1], label='K')
    
    plt.savefig("prod_v3_anomalies.png")
    
    # Zoom into a "noisy" region (e.g. North Atlantic/Southern Ocean)
    plt.figure(figsize=(8,8))
    plt.imshow(noise[30:45, 10:40], origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
    plt.title("Zoom: Grid-Scale Noise (N. Atlantic region)")
    plt.colorbar()
    plt.savefig("prod_v3_anomalies_zoom.png")
    
    print(f"Global RMS Noise: {jnp.sqrt(jnp.mean(noise**2)):.4f} K")

if __name__ == "__main__":
    analyze_noise()