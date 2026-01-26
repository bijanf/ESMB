
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from chronos_esm import io as model_io

def compute_roughness(field):
    """Compute spatial roughness (variance of Laplacian)."""
    # Simple discrete Laplacian
    lap = (
        jnp.roll(field, 1, axis=0) + jnp.roll(field, -1, axis=0) +
        jnp.roll(field, 1, axis=1) + jnp.roll(field, -1, axis=1) -
        4 * field
    )
    return jnp.mean(lap**2)

def check_latest_state():
    output_dir = Path("outputs/production_control")
    # Find latest mean file
    mean_files = sorted(list(output_dir.glob("mean_*.nc")))
    if not mean_files:
        print("No output files found yet.")
        return

    latest_file = mean_files[-1]
    print(f"Analyzing {latest_file}...")

    state = model_io.load_state_from_netcdf(latest_file)
    
    # Check SST Roughness (Composite - likely high due to coastlines)
    sst = state.fluxes.sst
    sst_roughness = compute_roughness(sst)
    print(f"Composite SST Roughness: {sst_roughness:.4f}")

    # Check Ocean Surface Temperature (Raw)
    ocean_temp = state.ocean.temp[0] # Surface
    
    # Mask out land (where velocity is exactly zero is a good proxy, or load mask)
    # Actually, let's load the mask to be sure.
    from chronos_esm import data
    mask = data.load_bathymetry_mask(nz=15)
    
    # Roughness only on Ocean points
    # We can't easily do a stencil on unstructured points, but we can zero out land contributions?
    # Better: Compute Laplacian everywhere, then mask the result.
    
    lap = (
        jnp.roll(ocean_temp, 1, axis=0) + jnp.roll(ocean_temp, -1, axis=0) +
        jnp.roll(ocean_temp, 1, axis=1) + jnp.roll(ocean_temp, -1, axis=1) -
        4 * ocean_temp
    )
    # Mask out land points in the Laplacian result (approximate)
    # If a point is ocean but neighbor is land, the Laplacian is valid (boundary gradient).
    # If a point is land, we don't care.
    lap_masked = jnp.where(mask, lap, 0.0)
    
    # Mean of squared laplacian over OCEAN points only
    ocean_roughness = jnp.sum(lap_masked**2) / jnp.sum(mask)
    
    print(f"Ocean Temp Roughness: {ocean_roughness:.4f}")
    
    # Check Max Ocean Temp
    max_ocn = jnp.max(jnp.where(mask, ocean_temp, -999.0))
    min_ocn = jnp.min(jnp.where(mask, ocean_temp, 999.0))
    print(f"Ocean Temp Range: {min_ocn:.2f} - {max_ocn:.2f} K")
    u = state.ocean.u[0] # Surface u
    u_roughness = compute_roughness(u)
    print(f"Surface U Roughness: {u_roughness:.4e}")

    # Check Temp Range
    print(f"SST Range: {jnp.min(sst):.2f} - {jnp.max(sst):.2f} K")
    print(f"T_mean: {jnp.mean(state.atmos.temp):.2f} K")

if __name__ == "__main__":
    check_latest_state()
