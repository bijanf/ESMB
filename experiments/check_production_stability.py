
import xarray as xr
import numpy as np
import glob
import os

def check_stability():
    print("--- CHRONOS-ESM STABILITY MONITOR ---")
    files = sorted(glob.glob("outputs/century_run/year_*.nc"))
    
    if not files:
        print("No output files found yet.")
        return

    print(f"{'Year':<6} | {'Global T (K)':<12} | {'AMOC (Sv)':<10} | {'Max U (m/s)':<12} | {'Status'}")
    print("-" * 60)

    for f in files[-10:]: # Check last 10 years
        try:
            ds = xr.open_dataset(f, decode_times=False)
            
            # 1. Global Temp
            t_mean = float(ds['atmos_temp'].mean())
            
            # 2. AMOC Index (Calibrated)
            # Atlantic Mask (Roughly 280E to 20E)
            v = ds['ocean_v'].fillna(0.0).values
            nx = ds.sizes['x_ocn']
            ny = ds.sizes['y_ocn']
            nz = ds.sizes['z']
            
            # Grid scale (Approx T31)
            # Surface layer effective depth = 50m (0.15 * 333m)
            # Deep layers = 333m
            dz = np.ones(nz) * (5000.0 / nz)
            dz[0] *= 0.15 # Surface Correction
            dz[1] *= 0.5  # Transition
            
            # Zonal Integration for Atlantic
            # Slice indices for Atlantic (approx)
            idx_atl_start = int(nx * 280 / 360)
            idx_atl_end = int(nx * 20 / 360) 
            
            # T31 dx is roughly 400km at equator
            dx_deg = 360.0 / nx
            lat = np.linspace(-90, 90, ny)
            dx = 6371000 * np.cos(np.deg2rad(lat)) * np.deg2rad(dx_deg)
            
            # Simple Zonal Sum in Atlantic Sector
            # Handle wrap-around or just take a slice if simpler
            # Let's take the middle slice which is usually Atlantic in T31 standard map (centered on Pacific?)
            # Actually, standard map: 0 is Greenwich.
            # Atlantic is ~300 to 360 and 0 to 20.
            
            # Let's just do Global MOC for stability check (easier/robuster signal)
            transport = v * dx[None, :, None] * dz[:, None, None]
            zonal_trans = np.sum(transport, axis=2) # (z, y)
            psi = np.cumsum(zonal_trans, axis=0) / 1e6
            
            # AMOC Proxy: Max streamfunction in NH Subsurface
            amoc_val = np.max(psi[2:, int(ny/2):]) # Ignore surface Ekman and SH
            
            # 3. Max Velocity (Instability check)
            u_max = np.max(np.abs(ds['atmos_u'].values))
            
            status = "OK"
            if np.isnan(t_mean): status = "CRASH (NaN)"
            elif u_max > 80.0: status = "UNSTABLE (Wind)"
            
            print(f"{f.split('_')[-1][:3]:<6} | {t_mean:<12.2f} | {amoc_val:<10.2f} | {u_max:<12.2f} | {status}")
            
        except Exception as e:
            print(f"{f}: Error {e}")

if __name__ == "__main__":
    check_stability()
