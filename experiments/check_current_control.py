
import xarray as xr
import numpy as np
from pathlib import Path

def check_control():
    path = Path("outputs/control_run_verified")
    files = sorted(list(path.glob("mean_*.nc")))
    if not files:
        print("No files found.")
        return

    last_file = files[-1]
    print(f"Analyzing {last_file}...")

    try:
        ds = None
        for eng in [None, "h5netcdf", "scipy", "netcdf4"]:
            try:
                ds = xr.open_dataset(last_file, decode_times=False, engine=eng)
                break
            except Exception:
                continue
        
        if ds is None:
            raise ValueError("Could not open file")

        
        # Grid parameters (T31)
        ny, nx = 48, 96
        R = 6371000.0
        dz = np.array([50.0, 50.0, 100.0, 100.0, 100.0, 200.0, 200.0, 300.0, 
                       400.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0])

        # SST
        sst_mean = float(ds.atmos_temp.mean())
        print(f"Global Mean Temp (Atmos): {sst_mean:.2f} K")

        # AMOC
        v = ds.ocean_v.values
        psi = np.zeros((len(dz), ny))
        lats = np.linspace(-90, 90, ny)
        
        for j in range(ny):
            lat = lats[j]
            dx = 2 * np.pi * R * np.cos(np.deg2rad(lat)) / nx
            v_zonal = np.sum(v[:, j, :] * dx, axis=1)
            current = 0.0
            for k in range(len(dz)):
                current += v_zonal[k] * dz[k]
                psi[k, j] = current
                
        amoc_max = np.max(np.abs(psi)) / 1e6
        print(f"AMOC Max: {amoc_max:.2f} Sv")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_control()
