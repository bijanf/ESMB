
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Config
OUTPUT_DIRS = [
    Path("outputs/long_run_fast"), # Y1-10
    Path("outputs/century_run")    # Y11+
]
PLOT_FILE = "timeseries_verification.png"

def calculate_metrics(ds, year):
    # Constants
    R_earth = 6.371e6
    cp_sw = 3985.0
    rho_sw = 1025.0
    
    # Grid Areas (Approx)
    ny, nx = ds.sizes['y_atm'], ds.sizes['x_atm']
    lat = np.linspace(-90, 90, ny)
    area = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, nx))
    area = area / area.sum() # Normalize weights
    
    # 1. TAS (Global Mean Atmos Temp)
    tas = (ds['atmos_temp'] * area).sum().values
    
    # 2. SST (Global Mean Ocean Surface Temp)
    # Ocean mask is implicit in values (0 if masked? No, mask is separate).
    # Assuming ocean everywhere for simplicity or use SST variable.
    sst = (ds['sst'] * area).sum().values
    
    # 3. OHC (Ocean Heat Content)
    # Integrate Temp * Rho * Cp * dV
    # We need depth profile. T31 has 15 levels.
    # We will compute Volume Mean Temp as proxy if depth metrics absent.
    # OHC ~ V * rho * cp * T_mean
    # Let's just track Mean Ocean Temp for now as OHC proxy.
    try:
        ohc = ds['ocean_temp'].mean().values # Volume mean
    except:
        ohc = np.nan
        
    # 4. AMOC / Global MOC
    # Integrate v * dx * dz from surface down
    try:
        # Standard Veros/MOM formulation
        # MOC(y, z) = - Integral_x ( Integral_z_bottom^z (v dx) dz )
        # or from Top: MOC(y, z) = Integral_x ( Integral_z_top^z (v dx) dz )
        
        # We need dx(y) and dz(k)
        # T31 Grid
        R = 6371000.0
        dlon = 360.0 / nx
        lat_rad = np.deg2rad(lat)
        dx = R * np.cos(lat_rad) * np.deg2rad(dlon) # (y,)
        
        # Vertical Thickness (Approximate for T31 15-level)
        # Check if dzw exists
        if 'dzt' in ds:
            dz = ds['dzt'].values
        else:
            # Fallback to hardcoded WOA15 levels if coords missing
            # 15 levels: Usually 50m top, increasing...
            # Let's try to infer from z coords
            z = ds['z'].values if 'z' in ds else np.arange(15)
            # Rough diff
            dz = np.gradient(z) # Very rough
            # Better: Use known standard levels 
            # 0-50, 50-100...
            # For now, let's assume standard dz=constant or simple scaling if not found
            # If dzt missing, check output again
            pass

        # Since output might not have dzt, let's compute Global MOC Sum(v) neglecting dz variation if desperate
        # BUT v is m/s. We need Transport (m3/s).
        # We MUST have dz.
        
        # Actually, let's look at the printed vars again. No dzt.
        # "ocean_v" (z, y, x).
        # "z" coord exists?
        
        # Update: We will assume 15 levels have specific thicknesses.
        # Standard: 50, 50, 50, ..., 500?
        # Let's perform a simple "Index" MOC (Sum v) * Scale Factor
        # Or better: Extract Z from ds.coords inside calculate_metrics
        
        # Implementation:
        v = ds['ocean_v'].fillna(0.0).values # (nz, ny, nx)
        
        # Atlantic Mask (Approximate)
        # T31 Longitudes: 0..360 approx
        lons = np.linspace(0, 360, nx, endpoint=False)
        # Atlantic Sector: ~290E to 360E  AND 0E to 20E (Eastern Atlantic)
        # Note: Ideally we use a land mask to separate Pacific, but standard longitude cut works for first order.
        atlantic_mask = (lons >= 280) | (lons <= 20) # Broad Atlantic
        
        # Expand mask to (1, 1, nx)
        mask_3d = np.zeros_like(v, dtype=bool)
        mask_3d[:, :, :] = atlantic_mask[None, None, :]
        
        # Zonal Sum: V_zonal_atl(z, y) = Sum(v * dx * mask)
        v_atl = np.where(mask_3d, v, 0.0)
        
        transport_per_lat_level = np.sum(v_atl, axis=2) * dx[None, :] # (nz, ny)
        
        # Vertical Integral (Cumulative Sum from Top)
        depths = ds['z'].values if 'z' in ds else np.linspace(0, 5000, 15)
        # Assume layers are diff(depths) or constant
        # Note: dz from config is best, but we don't have config obj here.
        # Fallback to simple integration with assumed 5000m depth
        total_depth = 5000.0
        n_layers = v.shape[0]
        # Use simple constant dz if depth info sparse
        # Actually, let's use the depths array if valid
        if len(depths) == n_layers:
             # Calculate layer thickness: Midpoints between depths?
             # Simple: Constant for now to avoid noise from bad dz
             dz_layers = np.ones(n_layers) * (total_depth / n_layers)
        else:
             dz_layers = np.ones(n_layers) * (total_depth / n_layers)

        transport_integrated = transport_per_lat_level * dz_layers[:, None] # (nz, ny)
        
        # Streamfunction (integrate from top)
        psi_ov = np.cumsum(transport_integrated, axis=0)
        
        # AMOC Index = Max in NH (Index 24-48 for T31), Atlantic Only
        # Latitude Mask: Equator (Index 24) to ~60N (Index 40)
        # Depth Mask: Below 500m (Index > 2) to exclude Ekman
        
        # Search range
        lat_min_idx = int(ny * 0.5) + 2 # Eq + a bit
        lat_max_idx = int(ny * 0.9)     # Polar circle
        depth_min_idx = 2               # Below surface
        
        # Find Max Overturning
        psi_subset = psi_ov[depth_min_idx:, lat_min_idx:lat_max_idx]
        amoc = np.max(np.abs(psi_subset)) * 1e-6 # Sv
        
    except Exception as e:
        print(f"AMOC Calc failed: {e}")
        amoc = np.nan
        
    return tas, sst, ohc, amoc

def plot_timeseries():
    years = []
    tass = []
    ssts = []
    ohcs = []
    amocs = []
    
    # Collect files
    files = []
    for d in OUTPUT_DIRS:
        if d.exists():
            files.extend(sorted(list(d.glob("year_*.nc"))))
            
    print(f"Found {len(files)} yearly files.")
    
    for f in files:
        try:
            year = int(f.name.split("_")[1].split(".")[0])
            ds = xr.open_dataset(f, decode_times=False)
            
            t, s, o, a = calculate_metrics(ds, year)
            
            years.append(year)
            tass.append(t)
            ssts.append(s)
            ohcs.append(o)
            amocs.append(a)
            ds.close()
            print(f"Processed Year {year}")
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # TAS
    axes[0, 0].plot(years, tass, 'r-o')
    axes[0, 0].set_title("Global Mean Surface Air Temperature (TAS)")
    axes[0, 0].set_ylabel("Temperature (K)")
    axes[0, 0].grid(True)
    
    # SST
    axes[0, 1].plot(years, ssts, 'b-o')
    axes[0, 1].set_title("Global Mean Sea Surface Temperature (SST)")
    axes[0, 1].set_ylabel("Temperature (K)")
    axes[0, 1].grid(True)
    
    # AMOC
    axes[1, 0].plot(years, amocs, 'g-o')
    axes[1, 0].set_title("Atlantic Meridional Overturning Circulation (AMOC)")
    axes[1, 0].set_ylabel("Transport (Sv)")
    axes[1, 0].grid(True)
    
    # OHC
    axes[1, 1].plot(years, ohcs, 'k-o')
    axes[1, 1].set_title("Mean Ocean Temperature (OHC Proxy)")
    axes[1, 1].set_ylabel("Temperature (K)")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    print(f"Plot saved to {PLOT_FILE}")

if __name__ == "__main__":
    plot_timeseries()
