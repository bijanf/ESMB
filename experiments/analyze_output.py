
import jax
import jax.numpy as jnp
import xarray as xr
import numpy as np
from pathlib import Path

def analyze_diagnostics():
    output_dir = Path("outputs/control_run")
    files = sorted(list(output_dir.glob("mean_*.nc")))
    
    if not files:
        print("No output files found.")
        return

    print(f"Analyzing {len(files)} monthly mean files...")
    
    amoc_series = []
    ice_extent_series = []
    temp_series = []
    months = []

    # Grid Parameters (Approximate for T31/T63)
    # We can't easily get exact dx(y) without config, but we can do a sum proxy
    # or try to import config. Let's try importing config.
    # Hardcoded Grid Parameters (from chronos_esm.config)
    # T31 / 15 Levels
    dz = np.array([50.0, 50.0, 100.0, 100.0, 100.0, 200.0, 200.0, 300.0, 
                   400.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0]) # 15 levels
    
    # Earth Radius
    R = 6371000.0
    nx = 96 # T31
    loss_factor = 2 * np.pi * R / nx 
    # dx ~= loss_factor * cos(lat)



    for f in files:
        ds = xr.open_dataset(f, decode_times=False)
        
        v = ds.ocean_v.values
        sst = ds.ocean_temp.values[0]
        
        print(f"DEBUG {f.name}: V_range=[{v.min():.4f}, {v.max():.4f}], V_mean={v.mean():.4e}")
        print(f"DEBUG {f.name}: SST_range=[{sst.min():.2f}, {sst.max():.2f}], SST_mean={sst.mean():.2f}")

        # Zonal Integral (Transport per depth layer)

        
        # Zonal Integral (Transport per depth layer)
        # Need lat coordinates. ds.ocean_v has coords? No, we used decode_times=False and raw load.
        # But we know ny=48 for T31? No ny=96 for T31 usually? 
        # Check standard config: ny is likely 48 or 96.
        # Let's assume uniform lat spacing from -90 to 90
        ny = v.shape[1] # 48 or 96
        lats = np.linspace(-90, 90, ny)
        
        psi = np.zeros_like(v[:, :, 0]) # (z, y)
        
        for j in range(ny):
            lat = lats[j]
            dx = 2 * np.pi * 6371000.0 * np.cos(np.deg2rad(lat)) / 96.0
            
            # Zonal sum of v * dx
            v_zonal_j = np.sum(v[:, j, :] * dx, axis=1) # (z,)
            
            # Cumulative vertical integral from surface (0) down
            current_transport = 0.0
            for k in range(len(dz)):
                current_transport += v_zonal_j[k] * dz[k]
                psi[k, j] = current_transport
             
        # Max Absolute Streamfunction (Global)
        psi_max = np.max(np.abs(psi)) / 1e6 # Sv
 
        
        # 2. Sea Ice Extent
        # We have ice_concentration (y, x)
        if 'ice_concentration' in ds:
            ice = ds.ice_concentration.values
            # Extent = Area where conc > 15%
            # Sum of cells > 0.15
            ice_extent = np.sum(ice > 0.15)
        else:
             # Fallback to SST proxy
             sst = ds.ocean_temp.values[0]
             ice_extent = np.sum(sst < 271.35)
        
        # 3. Global Temp
        t_mean = float(np.mean(ds.atmos_temp.values))
        
        amoc_series.append(psi_max)
        ice_extent_series.append(ice_extent)
        temp_series.append(t_mean)
        months.append(f.name.replace("mean_", "").replace(".nc", ""))
        
        ds.close()

    print(f"{'Month':<10} | {'AMOC (~Sv)':<12} | {'Ice(Cells)':<10} | {'T_Atmos':<10}")
    print("-" * 55)
    for i in range(len(files)):
        print(f"{months[i]:<10} | {amoc_series[i]:.2f}         | {ice_extent_series[i]:<10} | {temp_series[i]:.2f}")


if __name__ == "__main__":
    analyze_diagnostics()
