import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm.ocean import diagnostics as ocean_diagnostics
import matplotlib.pyplot as plt
import numpy as np

def analyze_amoc():
    output_dir = Path("outputs/production_control")
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    
    if not mean_files:
        print("No output files found.")
        return

    print(f"Found {len(mean_files)} monthly files. Calculating AMOC evolution...")

    months = []
    amoc_indices = []
    
    # process every file to get the full time series
    for i, f in enumerate(mean_files):
        if i % 12 == 0: print(f"Processing year {i//12}...")
        
        try:
            state = model_io.load_state_from_netcdf(f)
            # AMOC Index: Max streamfunction at 30N (approx index)
            # We can use the helper in diagnostics
            amoc_val = ocean_diagnostics.compute_amoc_index(state.ocean)
            
            # Extract month number from filename
            m = int(f.name.split("_")[1].split(".")[0])
            
            months.append(m)
            amoc_indices.append(float(amoc_val))
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # Plot Time Series
    plt.figure(figsize=(12, 6))
    plt.plot(months, amoc_indices, 'b-', linewidth=1)
    plt.xlabel('Month')
    plt.ylabel('AMOC Index (Sv)')
    plt.title('AMOC Evolution (100 Years)')
    plt.grid(True, alpha=0.3)
    plt.savefig("final_amoc_timeseries.png")
    print("Saved final_amoc_timeseries.png")
    
    # Plot Final Decade Mean Structure
    print("Calculating final decade mean MOC structure...")
    last_120 = mean_files[-120:]
    sum_moc = None
    
    for f in last_120:
        state = model_io.load_state_from_netcdf(f)
        moc = ocean_diagnostics.compute_moc(state.ocean)
        if sum_moc is None:
            sum_moc = np.array(moc)
        else:
            sum_moc += np.array(moc)
            
    mean_moc = sum_moc / len(last_120)
    
    plt.figure(figsize=(10, 6))
    lat = np.linspace(-90, 90, mean_moc.shape[1])
    depth = np.linspace(0, 5000, mean_moc.shape[0])
    
    # Plot MOC (Sv)
    limit = max(abs(np.min(mean_moc)), abs(np.max(mean_moc)), 5.0)
    levels = np.linspace(-limit, limit, 21)
    
    plt.contourf(lat, depth, mean_moc, levels=levels, cmap='RdBu_r', extend='both')
    plt.colorbar(label='MOC (Sv)')
    plt.contour(lat, depth, mean_moc, levels=levels, colors='k', linewidths=0.5, alpha=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel('Latitude')
    plt.ylabel('Depth (m)')
    plt.title('Mean Meridional Overturning Circulation (Final 10 Years)')
    plt.savefig("final_moc_structure.png")
    print("Saved final_moc_structure.png")

if __name__ == "__main__":
    analyze_amoc()