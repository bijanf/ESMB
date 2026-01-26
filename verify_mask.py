
import sys
import os

# Add parent path
sys.path.insert(0, os.getcwd())

from chronos_esm import data
import jax.numpy as jnp
import numpy as np

def verify():
    print("Loading bathymetry mask...")
    try:
        mask = data.load_bathymetry_mask(nz=15)
        print(f"Mask shape: {mask.shape}")
        print(f"Mask dtype: {mask.dtype}")
        
        mask_np = np.array(mask)
        n_ocean = np.sum(mask_np > 0.5)
        n_land = np.sum(mask_np <= 0.5)
        total = mask_np.size
        
        print(f"Ocean points: {n_ocean} ({n_ocean/total*100:.2f}%)")
        print(f"Land points: {n_land} ({n_land/total*100:.2f}%)")
        
        if n_land == 0:
            print("ERROR: No land points found! Mask is all Ocean.")
        else:
            print("Mask seems to contain land.")
            
    except Exception as e:
        print(f"Error loading mask: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
