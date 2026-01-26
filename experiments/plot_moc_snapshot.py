
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm.ocean import diagnostics as ocean_diagnostics
from chronos_esm import data
import matplotlib.pyplot as plt

def plot_moc():
    output_dir = Path("outputs/production_control")
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    
    if not mean_files:
        print("No output files found.")
        return

    latest_file = mean_files[-1]
    print(f"Plotting MOC for {latest_file}...")
    state = model_io.load_state_from_netcdf(latest_file)
    
    moc = ocean_diagnostics.compute_moc(state.ocean)
    
    plt.figure(figsize=(10, 6))
    lat = jnp.linspace(-90, 90, moc.shape[1])
    depth = jnp.linspace(0, 5000, moc.shape[0])
    
    # Flip depth for plotting
    plt.contourf(lat, depth, moc, levels=20, cmap='RdBu_r')
    plt.colorbar(label='Streamfunction (Sv)')
    plt.gca().invert_yaxis()
    plt.xlabel('Latitude')
    plt.ylabel('Depth (m)')
    plt.title(f'Global MOC - {latest_file.name}')
    
    plt.savefig("moc_snapshot.png")
    print("Saved plot to moc_snapshot.png")

if __name__ == "__main__":
    plot_moc()
