
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm import data
import numpy as np

def inspect_stats():
    output_dir = Path("outputs/production_control")
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    if not mean_files: return

    f = mean_files[-1]
    print(f"Inspecting {f.name}...")
    state = model_io.load_state_from_netcdf(f)
    
    # 1. SST
    sst = np.array(state.fluxes.sst)
    mask = data.load_bathymetry_mask(nz=15)
    sst_ocn = sst[mask]
    print(f"SST (Ocean): {sst_ocn.min():.2f} - {sst_ocn.max():.2f} K (Mean {sst_ocn.mean():.2f})")
    
    # 2. Ice
    ice = np.array(state.ice.concentration)
    print(f"Ice Conc: {ice.min():.2f} - {ice.max():.2f}")
    
    # 3. Atmos Temp
    ta = np.array(state.atmos.temp)
    print(f"Atmos Temp: {ta.min():.2f} - {ta.max():.2f} K")
    
    # 4. Precip
    precip = np.array(state.fluxes.precip) * 86400
    print(f"Precip (mm/day): {precip.min():.2f} - {precip.max():.2f} (Mean {precip.mean():.2f})")
    
    # 5. Wind Stress
    tau = np.array(state.fluxes.wind_stress_x)
    print(f"Tau X: {tau.min():.2e} - {tau.max():.2e} Pa")
    
    # 6. Salinity
    sss = np.array(state.ocean.salt[0])[mask]
    print(f"SSS: {sss.min():.2f} - {sss.max():.2f} psu")

if __name__ == "__main__":
    inspect_stats()
