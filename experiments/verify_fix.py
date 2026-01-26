import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm import data

def analyze_interior_noise():
    output_dir = Path("outputs/production_control")
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    if not mean_files: return

    latest_file = mean_files[-1]
    print(f"Analyzing {latest_file}...")
    state = model_io.load_state_from_netcdf(latest_file)
    sst = state.fluxes.sst
    
    # Load Mask
    mask = data.load_bathymetry_mask(nz=15)
    
    # Erode Mask (Remove coasts)
    # A point is Interior Ocean if it AND all 4 neighbors are Ocean
    mask_n = jnp.roll(mask, 1, axis=0)
    mask_s = jnp.roll(mask, -1, axis=0)
    mask_e = jnp.roll(mask, 1, axis=1)
    mask_w = jnp.roll(mask, -1, axis=1)
    
    interior_mask = mask & mask_n & mask_s & mask_e & mask_w
    
    # Calculate Noise (High-pass)
    smoothed = (jnp.roll(sst, 1, 0) + jnp.roll(sst, -1, 0) + jnp.roll(sst, 1, 1) + jnp.roll(sst, -1, 1)) / 4.0
    noise = sst - smoothed
    
    # Mask Noise
    noise_interior = jnp.where(interior_mask, noise, 0.0)
    
    rms_noise = jnp.sqrt(jnp.sum(noise_interior**2) / jnp.sum(interior_mask))
    print(f"Interior Ocean RMS Noise: {rms_noise:.4f} K")
    
    # Also check Land Noise
    land_mask = ~mask
    # Erode Land Mask? 
    # Let's just check full land
    noise_land = jnp.where(land_mask, noise, 0.0)
    rms_noise_land = jnp.sqrt(jnp.sum(noise_land**2) / jnp.sum(land_mask))
    print(f"Land RMS Noise: {rms_noise_land:.4f} K")

if __name__ == "__main__":
    analyze_interior_noise()