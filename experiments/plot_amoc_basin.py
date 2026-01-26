
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io
from chronos_esm import data
from chronos_esm.config import EARTH_RADIUS, OCEAN_GRID
import matplotlib.pyplot as plt
import numpy as np

def compute_basin_moc(state, mask, basin_lon_min, basin_lon_max):
    """
    Compute MOC restricted to a longitude range (Basin).
    """
    # 1. Create Basin Mask
    nx = OCEAN_GRID.nlon
    ny = OCEAN_GRID.nlat
    
    lon = np.linspace(-180, 180, nx, endpoint=False)
    lat = np.linspace(-90, 90, ny)
    
    # Broadcast lon to (ny, nx)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Select Atlantic: approx -80 to 0 (simplified)
    # Refined: 
    # S. Atlantic: -70 to 20
    # N. Atlantic: -80 to 0
    # We'll use a simple box for now: -80 to 10
    
    basin_mask = (lon_grid >= basin_lon_min) & (lon_grid <= basin_lon_max) & mask
    
    # 2. Integrate V over X (masked)
    v = np.array(state.ocean.v)
    
    # dx varies with latitude
    lat_rad = np.deg2rad(lat)
    dx = 2 * np.pi * EARTH_RADIUS * np.cos(lat_rad) / nx
    # Broadcast dx to (ny, nx)
    dx_grid = np.tile(dx[:, None], (1, nx))
    
    # V transport (m^2/s)
    v_trans = v * dx_grid[None, :, :]
    
    # Apply Basin Mask
    v_basin = np.where(basin_mask[None, :, :], v_trans, 0.0)
    
    # Sum over Lon
    v_zonal = np.sum(v_basin, axis=2) # (nz, ny)
    
    # 3. Integrate vertically (from top or bottom)
    # Depth = 5000m, 15 levels -> dz = 333.33m
    dz = 5000.0 / 15.0
    
    # Streamfunction Psi(y, z) = - Integral_z^0 v dz (Top Down)
    # Or Psi(y, z) = Integral_{-H}^z v dz (Bottom Up)
    # We use Top Down for standard MOC
    # cumsum axis 0
    
    moc = -np.cumsum(v_zonal * dz, axis=0)
    
    return moc / 1.0e6 # Sv

def plot_amoc_basin():
    output_dir = Path("outputs/production_control")
    results_dir = Path("analysis_results/prod_v6")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    if not mean_files:
        print("No output files found.")
        return

    # Load Final Year (Average last 12 months)
    print("Calculating Annual Mean for Final Year...")
    last_12 = mean_files[-12:]
    
    # Load one to get shape
    s0 = model_io.load_state_from_netcdf(last_12[0])
    sum_v = np.zeros_like(s0.ocean.v)
    
    for f in last_12:
        s = model_io.load_state_from_netcdf(f)
        sum_v += np.array(s.ocean.v)
        
    avg_v = sum_v / 12.0
    
    # Create a synthetic state with avg_v
    state_avg = s0._replace(ocean=s0.ocean._replace(v=jnp.array(avg_v)))
    
    mask = data.load_bathymetry_mask(nz=15)
    
    # 1. Global MOC
    print("Computing Global MOC...")
    moc_global = compute_basin_moc(state_avg, mask, -180, 180)
    
    # 2. Atlantic MOC (Approx: 80W to 10E)
    print("Computing Atlantic MOC...")
    moc_atl = compute_basin_moc(state_avg, mask, -80, 10)
    
    # 3. Pacific MOC (Approx: 120E to 80W -> 120 to 180 + -180 to -80?)
    # Easier: Global - Atlantic? Or specific box
    # Let's just plot Global and Atlantic
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    lat = np.linspace(-90, 90, moc_global.shape[1])
    depth = np.linspace(0, 5000, moc_global.shape[0])
    
    # Levels
    limit = max(np.abs(moc_global).max(), np.abs(moc_atl).max(), 5.0)
    levels = np.linspace(-limit, limit, 21)
    
    # Global
    im1 = axes[0].contourf(lat, depth, moc_global, levels=levels, cmap='RdBu_r', extend='both')
    axes[0].contour(lat, depth, moc_global, levels=levels, colors='k', linewidths=0.5, alpha=0.5)
    axes[0].set_title("Global MOC (Year 100)")
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Depth (m)")
    axes[0].set_xlabel("Latitude")
    plt.colorbar(im1, ax=axes[0], label='Sv')
    
    # Atlantic
    im2 = axes[1].contourf(lat, depth, moc_atl, levels=levels, cmap='RdBu_r', extend='both')
    axes[1].contour(lat, depth, moc_atl, levels=levels, colors='k', linewidths=0.5, alpha=0.5)
    axes[1].set_title("Atlantic MOC (Approx. 80W-10E)")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Latitude")
    plt.colorbar(im2, ax=axes[1], label='Sv')
    
    plt.tight_layout()
    plt.savefig(results_dir / "final_amoc_structure_v2.png")
    print(f"Saved {results_dir / 'final_amoc_structure_v2.png'}")

if __name__ == "__main__":
    plot_amoc_basin()
