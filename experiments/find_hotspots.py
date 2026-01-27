
import xarray as xr
import numpy as np
import sys

def find_hotspots(path):
    print(f"--- Diagnosing {path} ---")
    ds = xr.open_dataset(path, decode_times=False)
    
    # 1. Velocity Hotspots
    u = ds['ocean_u'].fillna(0.0).values
    v = ds['ocean_v'].fillna(0.0).values
    speed = np.sqrt(u**2 + v**2)
    
    max_speed = np.max(speed)
    argmax_speed = np.unravel_index(np.argmax(speed), speed.shape)
    
    print(f"Max Ocean Speed: {max_speed:.4f} m/s")
    print(f"Location (z, y, x): {argmax_speed}")
    print(f"U at Max: {u[argmax_speed]:.4f}, V at Max: {v[argmax_speed]:.4f}")
    
    # Check neighborhood of max speed
    z, y, x = argmax_speed
    if y > 0 and y < speed.shape[1]-1 and x > 0 and x < speed.shape[2]-1:
        print("Slice around Max Speed:")
        print(speed[z, y-1:y+2, x-1:x+2])
    
    # 2. Temperature Extremes
    t = ds['ocean_temp'].fillna(0.0).values
    print(f"Temp Range: {np.min(t):.2f} to {np.max(t):.2f} K")
    
    if np.min(t) < 271.0:
        print("WARNING: Supercooled water detected!")
        
    # 3. Check for Checkerboard (2*dt oscillations)
    # Simple Laplacian check normalized by field variance
    # lap = |4T - neighbors|
    lap = np.abs(4*t[0] - np.roll(t[0],1,0) - np.roll(t[0],-1,0) - np.roll(t[0],1,1) - np.roll(t[0],-1,1))
    noise_max = np.max(lap)
    noise_mean = np.mean(lap)
    print(f"Grid Noise (Laplacian T): Max={noise_max:.4f}, Mean={noise_mean:.4f}")
    
    if noise_max > 5.0:
        print("!! HIGH NOISE DETECTED !! Probability of crash: HIGH")
    
    # 4. Atmos Winds
    ua = ds['atmos_u'].values
    va = ds['atmos_v'].values
    w_speed = np.sqrt(ua**2 + va**2)
    print(f"Max Wind Speed: {np.max(w_speed):.2f} m/s")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        find_hotspots(sys.argv[1])
    else:
        find_hotspots("outputs/century_run/year_038.nc")
