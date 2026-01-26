
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from chronos_esm.ocean import diagnostics

def analyze_current_run():
    # Find all mean files
    files = sorted(glob.glob('outputs/control_run_prod_v1/mean_*.nc'))
    if not files:
        print("No output files found!")
        return

    print(f"Found {len(files)} monthly files.")
    
    # Process in chunks or individually to avoid memory issues (though specific to this env)
    # We just want time series, so we can load minimal data
    
    times = []
    temps = []
    amocs = []
    
    for f in files:
        ds = xr.open_dataset(f, decode_times=False)
        
        # Time
        # Assuming filename mean_XXXX.nc corresponds to month XXXX
        month = int(os.path.basename(f).split('_')[1].split('.')[0])
        times.append(month / 12.0) # Year
        
        # Temp
        # Global mean T_air (area weighted if needed, but simple mean for quick check)
        # Using atmos temp
        t_global = ds.atmos_temp.mean().values
        temps.append(t_global)
        
        # AMOC
        # Need to reconstruct ocean state object or just compute directly
        # diagnostics.compute_amoc_index expects an OceanState namedtuple, which is tricky to reconstruct from xarray
        # But we can look at veros_driver / diagnostics code.
        # compute_amoc_index(ocean_state) -> looks at v velocity
        # Let's just implement the AMOC calculation here manually for the xarray dataset
        
        # AMOC at 26N (approx index 35 for 48 lat T31 grid?)
        # T31 Lat: -90 to 90 with 48 points. dy ~ 3.75 deg. 
        # 26N is roughly index 30-31? (-90 + 3.75*i = 26 -> 3.75*i = 116 -> i=30.9)
        # We need to integrate v * dx * dz zonal and vertical
        
        # Simple max of streamfunction is better.
        # But for quick check, let's try to reuse the code if possible or simplify.
        
        # Converting xarray to simple object with .u, .v attributes for the diagnostics function
        class SimpleOcean:
            def __init__(self, u, v, w):
                self.u = u
                self.v = v
                self.w = w
        
        # Load v (needs to correspond to correct time)
        # outputs have ocean_u, ocean_v
        v = ds.ocean_v.values # (nz, ny, nx)
        u = ds.ocean_u.values
        # w might not be saved? Check file.
        # If w missing, calculate from continuity? Or maybe we just skip and do a rough estimate.
        
        # Actually, let's look at `check_amoc_quick.py` strategy if exists.
        # Or just compute max streamfunction (psi)
        # psi(z, y) = sum_x(v * dx) * dz
        # cumsum from bottom
        
        # Grid info (hardcoded for T31/15L for speed)
        dx = 2 * np.pi * 6371000 * np.cos(np.deg2rad(np.linspace(-90, 90, 48))) / 96
        # Broadcast dx to (nz, ny, nx) if needed, but we sum over x first
        
        # Zonal integration of v (Sum over x)
        # v: (z, y, x)
        # mask is needed!
        
        # Let's use the actual diagnostics if I can import it properly, 
        # but it requires JAX arrays.
        # Let's do a numpy calculation.
        
        v_transport = v * 295000 * 333 # Rough dx, dz
        # This is too rough.
        
        # Better: Reuse code we saw in `chronos_esm/ocean/diagnostics.py`?
        pass

    # RE-STRATEGY: Use the existing code logic for AMOC
    # AMOC is max of overturning streamfunction in Atlantic.
    # We will just plot 'AMOC Index' if available or calculate it properly.
    # Wait, the log file PRINTS the AMOC index at the end of run? No, only monthly logs print T_mean.
    
    # Let's calculate AMOC properly for the LAST file only to see current state, 
    # and maybe estimate history if fast.
    
    print(f"Analyzing {len(files)} files...")
    
    # Extract AMOC from each file? 
    # It seems verify heavy to compute full AMOC for all history.
    # Let's do it for the last 5 files.
    
    last_files = files[-5:]
    
    current_amoc = 0.0
    
    # We need to compute it.
    # Let's use a very simplified calculation for speed.
    
    # Load last file
    last_ds = xr.open_dataset(files[-1], decode_times=False)
    
    # Global Mean Temp History
    t_series = []
    
    # We can iterate fast for T
    for f in files:
         d = xr.open_dataset(f, decode_times=False)
         t_series.append(d.atmos_temp.mean())
         
    plt.figure()
    plt.plot(np.array(times), np.array(t_series))
    plt.xlabel('Year')
    plt.ylabel('Global Mean T (K)')
    plt.title('Production Run Temperature')
    plt.grid(True)
    plt.savefig('prod_temp.png')
    
    print(f"Latest Temperature: {t_series[-1]:.2f} K")
    
    # AMOC Calculation for LAST Snapshot
    # Load V
    v = last_ds.ocean_v.fillna(0.0).values # (z, y, x)
    
    # Dz approx 5000/15
    dz = 5000.0/15.0
    
    # Dx varies with latitude
    lats = last_ds.y_ocn.values
    dx = 2 * np.pi * 6371000 * np.cos(np.deg2rad(lats)) / 96.0
    
    # Zonal Sum (Atlantic only?)
    # Simply sum all x for global MOC as proxy if we don't have basin mask handy
    # Ideally we restrict to Atlantic indices (e.g. x index 60-90?)
    # For now, Global MOC is okay proxy for AMOC style behavior check
    
    v_zonal = np.sum(v, axis=2) # (z, y)
    
    # Transport = v_zonal * dx * dz
    # But dx depends on y.
    
    psi = np.zeros_like(v_zonal)
    
    # Integrate from top (z=0) down? or bottom up
    # Psi(z) = Int(v dz dx)
    
    current_psi = 0.0
    for k in range(v.shape[0]): # 0 is top
        # Transport at this layer k
        # T(y) = v_zonal(k, y) * dx(y) * dz
        trans = v_zonal[k, :] * dx * dz
        current_psi -= trans # Minus because usually integrated from bottom? 
                             # Or Top-Down: Psi(z) = Psi(z-1) - V_transport(z)
                             # Sign convention varies.
        psi[k, :] = current_psi
        
    # Max Psi in Sv
    max_psi = np.max(psi) / 1e6
    print(f"Estimated Max Overturning (Global proxy for AMOC): {max_psi:.2f} Sv")

if __name__ == "__main__":
    analyze_current_run()
