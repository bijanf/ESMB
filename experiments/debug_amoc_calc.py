
import xarray as xr
import numpy as np

def debug_amoc(file_path):
    ds = xr.open_dataset(file_path, decode_times=False)
    
    # Grid
    nx = ds.sizes['x_ocn']
    ny = ds.sizes['y_ocn']
    nz = ds.sizes['z']
    
    print(f"Grid: {nx}x{ny}x{nz}")
    
    # 1. DX
    R = 6371000.0
    lat = np.linspace(-90, 90, ny)
    dlon = 360.0 / nx
    dx = R * np.cos(np.deg2rad(lat)) * np.deg2rad(dlon)
    print(f"DX at Equator: {dx[int(ny/2)]/1000:.1f} km")
    print(f"DX at 45N: {dx[int(ny*0.75)]/1000:.1f} km")
    
    # 2. DZ (Assumed Linear)
    total_depth = 5000.0
    dz = total_depth / nz
    print(f"DZ per layer: {dz:.1f} m")
    
    # 3. V
    v = ds['ocean_v'].fillna(0.0).values
    v_atl = v.copy()
    
    # Mask Atlantic (Simplified)
    lons = np.linspace(0, 360, nx, endpoint=False)
    mask = (lons >= 280) | (lons <= 20)
    v_atl[:, :, ~mask] = 0.0
    
    print(f"V_Atl Max: {np.max(v_atl):.4f} m/s")
    print(f"V_Atl Min: {np.min(v_atl):.4f} m/s")
    
    # 4. Integrate
    # Transport per cell = v * dx * dz [m3/s]
    transport = v_atl * dx[None, :, None] * dz
    
    # Zonal Sum [m3/s]
    transport_zonal = np.sum(transport, axis=2) # (nz, ny)
    
    # Vertical Integral from Top (Streamfunction)
    psi = np.cumsum(transport_zonal, axis=0)
    
    # Convert to Sv
    psi_sv = psi / 1e6
    
    # Find AMOC Max (NH, Subsurface)
    # y index > eq (24), z index > 2
    psi_nh = psi_sv[2:, 24:]
    amoc_max = np.max(np.abs(psi_nh))
    
    print(f"Calculated AMOC Max: {amoc_max:.2f} Sv")
    
    # Check if we were using Sum of Abs or something?
    # No, cumsum is standard.
    
    return amoc_max

if __name__ == "__main__":
    debug_amoc("outputs/century_run/year_038.nc")
