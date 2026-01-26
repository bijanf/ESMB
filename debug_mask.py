
import sys
import os
import jax.numpy as jnp
import numpy as np

# Add parent directory to path
sys.path.append(os.getcwd())

from chronos_esm import data

def check_mask():
    print("Loading Bathymetry Mask...")
    mask = data.load_bathymetry_mask()
    
    mean_val = jnp.mean(mask)
    print(f"Mask Mean: {mean_val}")
    print(f"Mask Min: {jnp.min(mask)}")
    print(f"Mask Max: {jnp.max(mask)}")
    
    if mean_val == 1.0:
        print("FAIL: Mask is All Ocean (1.0)")
    elif mean_val == 0.0:
        print("FAIL: Mask is All Land (0.0)")
    else:
        print("SUCCESS: Mask has mixed Land/Ocean.")

if __name__ == "__main__":
    check_mask()
