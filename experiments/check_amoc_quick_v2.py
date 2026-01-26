
import sys
import os
sys.path.append(os.getcwd()) # Fix Import
import jax.numpy as jnp
from chronos_esm import main, data
from chronos_esm.ocean import diagnostics

def check_amoc(path):
    print(f"Checking AMOC for {path}...")
    ds = data.load_run_output(path)
    # Reconstruct state object (simplified)
    # We just need ocean.u, ocean.v, ocean.w for MOC?
    # Actually diagnostics.compute_moc takes 'ocean_state'.
    # We can mock it.
    
    class MockOcean:
        def __init__(self, ds):
            self.u = jnp.array(ds['u'])
            self.v = jnp.array(ds['v'])
            self.w = jnp.array(ds['w'])
            self.mask = jnp.array(ds['mask'])
            
    ocean = MockOcean(ds)
    moc = diagnostics.compute_moc(ocean)
    
    # Get Max AMOC at 30N (approx index)
    nlat = moc.shape[1]
    lat_idx_30n = int((30.0 + 90.0) / 180.0 * nlat)
    
    # Scan depth/lat for global max
    max_amoc = jnp.max(moc)
    amoc_30n = jnp.max(moc[:, lat_idx_30n])
    
    print(f"Global Max MOC: {max_amoc:.2f} Sv")
    print(f"AMOC at 30N: {amoc_30n:.2f} Sv")

if __name__ == "__main__":
    check_amoc(sys.argv[1])
