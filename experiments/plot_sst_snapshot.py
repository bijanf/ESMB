
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm import data
import matplotlib.pyplot as plt

def plot_sst():
    output_dir = Path("outputs/production_control")
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    
    if not mean_files:
        print("No output files found.")
        return

    latest_file = mean_files[-1]
    print(f"Plotting SST for {latest_file}...")
    state = model_io.load_state_from_netcdf(latest_file)
    
    sst = state.fluxes.sst
    
    plt.figure(figsize=(12, 6))
    plt.imshow(sst, origin='lower', cmap='RdYlBu_r')
    plt.colorbar(label='SST (K)')
    plt.title(f'Sea Surface Temperature - {latest_file.name}')
    
    plt.savefig("sst_snapshot.png")
    print("Saved plot to sst_snapshot.png")

if __name__ == "__main__":
    plot_sst()
