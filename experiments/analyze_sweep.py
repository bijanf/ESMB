
import xarray as xr
import numpy as np
import os
from pathlib import Path

def analyze_sweep():
    """
    Analyzes all sweep results and prints a comparison table.
    """
    base_dir = Path("outputs")
    
    # targets
    TARGET_AMOC = 17.0 # Sv
    TARGET_SST = 288.0 # K (15C)
    
    # Find all sweep directories
    sweep_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and "control_run_sweep_" in d.name])
    
    print(f"{'Run Configuration':<30} | {'Status':<10} | {'AMOC (Sv)':<10} | {'SST (K)':<10} | {'Score':<10}")
    print("-" * 90)
    
    results = []
    
    # Grid Parameters (Approximation for T31)
    # T31: 96x48
    nx = 96
    ny = 48
    R = 6371000.0
    
    # Depths (from analyze_output.py)
    dz = np.array([50.0, 50.0, 100.0, 100.0, 100.0, 200.0, 200.0, 300.0, 
                   400.0, 500.0, 500.0, 500.0, 500.0, 500.0, 500.0])
    
    for d in sweep_dirs:
        # Check for latest mean file
        files = sorted(list(d.glob("mean_*.nc")))
        if not files:
            print(f"{d.name:<30} | {'EMPTY':<10} | {'-':<10} | {'-':<10} | {'-':<10}")
            continue
            
        last_file = files[-1]
        
        try:
            ds = None
            # Try multiple engines
            for eng in [None, "h5netcdf", "scipy", "netcdf4"]:
                try:
                    ds = xr.open_dataset(last_file, decode_times=False, engine=eng)
                    break
                except Exception:
                    continue
            
            if ds is None:
                raise ValueError("Could not open file with any available engine (netcdf4/h5netcdf/scipy).")

            # 1. Global Mean SST            
            # 1. Global Mean SST
            sst_mean = float(ds.atmos_temp.mean()) # Global mean surface air temp actually (best proxy for 2m) or use fluxes.sst
            # Use atmos_temp as "Global Mean Temp" target
            
            # 2. AMOC
            v = ds.ocean_v.values
            if v.shape[1] != ny:
                # Mismatch grid?
                pass
                
            # Compute Streamfunction
            psi = np.zeros((len(dz), ny))
            lats = np.linspace(-90, 90, ny)
            
            for j in range(ny):
                lat = lats[j]
                dx = 2 * np.pi * R * np.cos(np.deg2rad(lat)) / nx
                v_zonal = np.sum(v[:, j, :] * dx, axis=1) # Sum over longitude
                
                # Integrate vertically
                current = 0.0
                for k in range(len(dz)):
                    current += v_zonal[k] * dz[k]
                    psi[k, j] = current
                    
            amoc_max = np.max(np.abs(psi)) / 1e6 # Sv
            
            # Score (Lower is better)
            score = abs(amoc_max - TARGET_AMOC) + abs(sst_mean - TARGET_SST)
            
            results.append({
                "name": d.name.replace("control_run_", ""),
                "amoc": amoc_max,
                "sst": sst_mean,
                "score": score,
                "file": last_file.name
            })
            
            ds.close()
            
        except Exception as e:
            print(f"{d.name:<30} | {'ERROR':<10} | {str(e)}")
            continue

    # Sort by Score
    results.sort(key=lambda x: x["score"])
    
    for r in results:
        print(f"{r['name']:<30} | {'DONE ('+r['file']+')':<10} | {r['amoc']:<10.2f} | {r['sst']:<10.2f} | {r['score']:<10.2f}")

    print("-" * 90)
    if results:
        best = results[0]
        print(f"Best Match: {best['name']} (AMOC={best['amoc']:.2f}, SST={best['sst']:.2f})")

if __name__ == "__main__":
    analyze_sweep()
