
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm.ocean import diagnostics as ocean_diagnostics
import matplotlib.pyplot as plt

def plot_amoc_timeseries():
    output_dir = Path("outputs/production_control")
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    
    if not mean_files:
        print("No output files found.")
        return

    months = []
    amocs = []
    
    print(f"Analyzing {len(mean_files)} files...")
    
    # Check every file
    for f in mean_files:
        try:
            m = int(f.name.split("_")[1].split(".")[0])
            state = model_io.load_state_from_netcdf(f)
            amoc = float(ocean_diagnostics.compute_amoc_index(state.ocean))
            months.append(m)
            amocs.append(amoc)
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    # Sort by month
    data = sorted(zip(months, amocs))
    months, amocs = zip(*data)
    
    # Filter: Start from Month 20
    start_idx = 0
    for i, m in enumerate(months):
        if m >= 20:
            start_idx = i
            break
            
    months = months[start_idx:]
    amocs = amocs[start_idx:]
            
    plt.figure(figsize=(10, 6))
    plt.plot(months, amocs, 'b-')
    plt.xlabel('Month')
    plt.ylabel('Global AMOC Index (Sv)')
    plt.title('AMOC Evolution (Spin-up Phase)')
    plt.grid(True)
    plt.savefig("production_amoc_timeseries_v2.png")
    print("Saved production_amoc_timeseries_v2.png")

if __name__ == "__main__":
    plot_amoc_timeseries()
