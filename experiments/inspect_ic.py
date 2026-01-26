
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm import data
import numpy as np

def inspect_ic():
    output_dir = Path("outputs/production_control")
    ic_file = output_dir / "state_0000.nc"
    
    if not ic_file.exists():
        print("No IC file found.")
        return

    print(f"Analyzing {ic_file}...")
    state = model_io.load_state_from_netcdf(ic_file)
    sst = state.fluxes.sst
    mask = data.load_bathymetry_mask(nz=15)
    
    sst_np = np.array(sst)
    mask_np = np.array(mask)
    
    ocean_temps = sst_np[mask_np]
    
    print(f"IC Ocean Temp: Min={ocean_temps.min():.2f}, Max={ocean_temps.max():.2f}, Mean={ocean_temps.mean():.2f}")

    if ocean_temps.max() < 290.0:
        print("WARNING: Initial Condition is too cold! Check data loading.")
    else:
        print("Initial Condition looks reasonable (Warm Pool exists).")

if __name__ == "__main__":
    inspect_ic()
