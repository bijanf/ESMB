
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm import data
from chronos_esm.ocean import diagnostics as ocean_diagnostics
import matplotlib.pyplot as plt
import numpy as np

def plot_final_results():
    output_dir = Path("outputs/production_control")
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    if not mean_files:
        print("No output files found.")
        return

    # Load Latest State
    latest_file = mean_files[-1]
    print(f"Plotting results for {latest_file}...")
    state = model_io.load_state_from_netcdf(latest_file)
    
    # Load Initial State for Drift Check
    ic_file = output_dir / "mean_0001.nc"
    if ic_file.exists():
        state_ic = model_io.load_state_from_netcdf(ic_file)
        sst_ic = state_ic.fluxes.sst
    else:
        sst_ic = None

    # 1. SST Map
    sst = state.fluxes.sst
    mask = data.load_bathymetry_mask(nz=15)
    sst_masked = np.where(mask, sst, np.nan)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(sst_masked, origin='lower', cmap='RdYlBu_r', vmin=270, vmax=315)
    plt.colorbar(label='SST (K)')
    plt.title(f'Sea Surface Temperature - Year {int(latest_file.name.split("_")[1].split(".")[0])/12:.1f}')
    plt.savefig("final_sst_map.png")
    
    # 2. Temperature Time Series
    months = []
    t_means = []
    for f in mean_files[::5]: # Every 5th file for speed
        m = int(f.name.split("_")[1].split(".")[0])
        s = model_io.load_state_from_netcdf(f)
        months.append(m)
        t_means.append(float(jnp.mean(s.atmos.temp)))
        
    plt.figure(figsize=(10, 5))
    plt.plot(months, t_means, 'r-')
    plt.xlabel('Month')
    plt.ylabel('Global Mean Temp (K)')
    plt.title('Global Temperature Evolution')
    plt.grid(True)
    plt.savefig("final_t_mean_timeseries.png")
    
    # 3. AMOC / MOC Streamfunction
    # Compute MOC
    # Use state.ocean.v (Meridional Velocity)
    moc = ocean_diagnostics.compute_moc(state.ocean)
    
    plt.figure(figsize=(10, 6))
    lat = jnp.linspace(-90, 90, moc.shape[1])
    depth = jnp.linspace(0, 5000, moc.shape[0])
    
    # Plot MOC (Sv)
    # Contour levels: -20 to 20 Sv
    levels = np.linspace(-25, 25, 21)
    plt.contourf(lat, depth, moc, levels=levels, cmap='RdBu_r', extend='both')
    plt.colorbar(label='MOC (Sv)')
    plt.gca().invert_yaxis()
    plt.xlabel('Latitude')
    plt.ylabel('Depth (m)')
    plt.title(f'Meridional Overturning Circulation - Year {int(latest_file.name.split("_")[1].split(".")[0])/12:.1f}')
    plt.savefig("final_moc_map.png")
    
    # 4. Surface Velocity Magnitude
    u_surf = state.ocean.u[0]
    v_surf = state.ocean.v[0]
    speed = jnp.sqrt(u_surf**2 + v_surf**2)
    speed_masked = np.where(mask, speed, np.nan)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(speed_masked, origin='lower', cmap='viridis', vmin=0, vmax=0.5)
    plt.colorbar(label='Speed (m/s)')
    plt.title('Surface Current Speed')
    plt.savefig("final_speed_map.png")

    print("Plots generated: final_sst_map.png, final_t_mean_timeseries.png, final_moc_map.png, final_speed_map.png")

if __name__ == "__main__":
    plot_final_results()
