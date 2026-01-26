
import xarray as xr
import numpy as np
import sys
import os

def check_amoc(filename):
    print(f"Calculating AMOC for {filename}...")
    try:
        ds = xr.open_dataset(filename, decode_times=False, engine='netcdf4')
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    if 'ocean_v' not in ds:
        print("ocean_v variable not found.")
        return

    v = ds['ocean_v'].values
    # If 4D (time, z, y, x), take first time step
    if v.ndim == 4:
        v = v[0]
    
    # Grid parameters (Approximate for T31)
    nz, ny, nx = v.shape
    EARTH_RADIUS = 6.371e6
    dz = 5000.0 / nz # 5000m depth / nz layers
    
    lat_vals = np.linspace(-90, 90, ny)
    lat_rad = np.deg2rad(lat_vals)
    
    # dx varies with latitude
    dx = 2 * np.pi * EARTH_RADIUS * np.cos(lat_rad) / nx
    dx = np.broadcast_to(dx[:, None], (ny, nx)) # (ny, nx)
    
    # Zonal Integration: Sum(v * dx) [m^2/s]
    # resulting shape (nz, ny)
    transport_zonal = np.sum(v * dx[None, :, :], axis=2)
    
    # Vertical Integration: Cumsum(transport * dz) [m^3/s]
    # We integrate from surface down (negative sign for streamfunction convention often used, 
    # but for MOC usually we want the overturning). 
    # MOC(y, z) = Integral_z^0 v dx dz'
    # Actually, usually defined as Integral_bottom^z v dx dz'.
    # If we integrate top-down: Psi(z) = - Integral_z^0 v dx dz'.
    
    moc = -np.cumsum(transport_zonal * dz, axis=0)
    
    # Convert to Sverdrups
    moc_sv = moc / 1.0e6
    
    # Find Max AMOC (typically in NH, e.g. 20N to 60N)
    # 20N -> index ~ ny * (20+90)/180
    idx_20n = int(ny * (20+90)/180)
    idx_60n = int(ny * (60+90)/180)
    
    # Restrict search to this latitude band and depth (usually valid below surface)
    moc_subset = moc_sv[:, idx_20n:idx_60n]
    amoc_max = np.max(moc_subset)
    
    print(f"AMOC Max (20N-60N): {amoc_max:.2f} Sv")
    
    # Also print profile at 30N
    idx_30n = int(ny * (30+90)/180)
    print(f"AMOC at ~30N (Profile Max): {np.max(moc_sv[:, idx_30n]):.2f} Sv")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_amoc(sys.argv[1])
    else:
        print("Usage: python check_amoc_quick.py <filename>")
