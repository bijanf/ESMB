
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm.ocean import diagnostics as ocean_diagnostics
from chronos_esm import data
import matplotlib.pyplot as plt

def compute_roughness(field, mask):
    lap = (
        jnp.roll(field, 1, axis=0) + jnp.roll(field, -1, axis=0) +
        jnp.roll(field, 1, axis=1) + jnp.roll(field, -1, axis=1) -
        4 * field
    )
    lap_masked = jnp.where(mask, lap, 0.0)
    return jnp.sum(lap_masked**2) / jnp.sum(mask)

def analyze_evolution():
    output_dir = Path("outputs/production_control")
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    
    if not mean_files:
        print("No output files found.")
        return

    mask = data.load_bathymetry_mask(nz=15)
    
    months = []
    roughnesses = []
    t_means = []
    amocs = []

    print(f"Analyzing {len(mean_files)} months...")
    
    # Analyze every 10th month to save time if there are many
    stride = max(1, len(mean_files) // 20)
    
    for i in range(0, len(mean_files), stride):
        f = mean_files[i]
        month = int(f.name.split("_")[1].split(".")[0])
        state = model_io.load_state_from_netcdf(f)
        
        rough = compute_roughness(state.ocean.temp[0], mask)
        t_mean = jnp.mean(state.atmos.temp)
        amoc = ocean_diagnostics.compute_amoc_index(state.ocean)
        
        months.append(month)
        roughnesses.append(float(rough))
        t_means.append(float(t_mean))
        amocs.append(float(amoc))
        print(f"Month {month:04d} | Roughness: {rough:8.2f} | T_mean: {t_mean:6.2f} | AMOC: {amoc:6.2f}")

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    axes[0].plot(months, roughnesses, 'r-o')
    axes[0].set_ylabel("Ocean Temp Roughness")
    axes[0].set_title("Model Stability & Evolution")
    
    axes[1].plot(months, t_means, 'b-o')
    axes[1].set_ylabel("Global Mean Temp (K)")
    
    axes[2].plot(months, amocs, 'g-o')
    axes[2].set_ylabel("AMOC Index (Sv)")
    axes[2].set_xlabel("Month")
    
    plt.tight_layout()
    plt.savefig("production_stability_check.png")
    print("Saved plot to production_stability_check.png")

if __name__ == "__main__":
    analyze_evolution()
