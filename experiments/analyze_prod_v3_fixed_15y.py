
import xarray as xr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def analyze_15y():
    # 1. Output Directory and Files
    data_dir = "outputs/control_run_prod_v3_fixed"
    files = sorted(glob.glob(os.path.join(data_dir, "mean_*.nc")))
    
    # Filter for first 182 files if more exist
    # files = files[:182] 
    
    if not files:
        print("No files found.")
        return

    print(f"Analyzing {len(files)} monthly files...")
    
    # 2. Storage for Time Series
    times = [] # Month index
    sat_mean = []
    sst_mean = []
    sss_mean = []
    amoc_idx = []
    
    # Constants
    EARTH_RADIUS = 6.371e6
    
    # Climatology Accumulator (Last 60 months / 5 years)
    clim_start_idx = max(0, len(files) - 60)
    sst_accum = None
    sss_accum = None
    n_accum = 0
    
    for i, f in enumerate(files):
        try:
            with xr.open_dataset(f, decode_times=False) as ds:
                # Temp variables
                t_sat = np.nan
                t_sst = np.nan
                t_sss = np.nan
                t_amoc = np.nan
                
                # SAT
                sat = ds.atmos_temp.values
                t_sat = np.mean(sat)
                
                # Ocean Fields
                if 'ocean_temp' in ds:
                    sst = ds.ocean_temp.isel(z=0).values - 273.15
                    t_sst = np.mean(sst)
                    
                    if i >= clim_start_idx:
                        if sst_accum is None: sst_accum = np.zeros_like(sst)
                        sst_accum += sst
                
                if 'ocean_salt' in ds:
                    sss = ds.ocean_salt.isel(z=0).values
                    t_sss = np.mean(sss)
                    
                    if i >= clim_start_idx:
                        if sss_accum is None: sss_accum = np.zeros_like(sss)
                        sss_accum += sss
                        n_accum += 1

                # AMOC Calculation
                if 'ocean_v' in ds:
                    v = ds.ocean_v.values
                    nz, ny, nx = v.shape
                    
                    # AMOC Logic
                    lat_vals = np.linspace(-90, 90, ny)
                    rad_lat = np.deg2rad(lat_vals)
                    dx_val = 2 * np.pi * EARTH_RADIUS * np.cos(rad_lat) / nx
                    # dx_val is (ny,)
                    # Broadcast to (1, ny, 1) for multiplication with v (nz, ny, nx)
                    transport = np.sum(v * dx_val[None, :, None], axis=2)
                    dz = 5000.0 / nz 
                    psi = -np.cumsum(transport * dz, axis=0)
                    psi_sv = psi / 1e6
                    
                    idx_30n = int((30 + 90)/180 * ny)
                    t_amoc = np.max(psi_sv[:, idx_30n])
                
                # Append all
                sat_mean.append(t_sat)
                sst_mean.append(t_sst)
                sss_mean.append(t_sss)
                amoc_idx.append(t_amoc)
                times.append(i+1)
                
        except Exception as e:
            print(f"Error reading {f}: {e}")
            # Ensure lists stay in sync if we continue, or just skip this time step
            # If we skip, we don't append to times, so it's fine.
            continue
                
        except Exception as e:
            print(f"Error reading {f}: {e}")
            break

    # 3. Plot Time Series
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # SAT & SST
    axes[0].plot(times, sat_mean, label='SAT (Global Mean)', color='tab:red')
    axes[0].plot(times, sst_mean, label='SST (Global Mean)', color='tab:orange', linestyle='--')
    axes[0].set_ylabel('Temperature (K / C)')
    axes[0].set_title('Global Mean Temperature')
    axes[0].legend()
    axes[0].grid(True)
    
    # SSS
    axes[1].plot(times, sss_mean, label='SSS (Global Mean)', color='tab:green')
    axes[1].set_ylabel('Salinity (PSU)')
    axes[1].set_title('Global Mean Sea Surface Salinity')
    axes[1].legend()
    axes[1].grid(True)
    
    # AMOC
    axes[2].plot(times, amoc_idx, label='AMOC @ 30N', color='tab:blue')
    axes[2].set_ylabel('Transport (Sv)')
    axes[2].set_xlabel('Simulation Month')
    axes[2].set_title('Atlantic Meridional Overturning Circulation (AMOC)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('prod_v3_fixed_15y_timeseries.png')
    print("Saved prod_v3_fixed_15y_timeseries.png")
    
    # 4. Plot Climatology Maps (Last 5 Years)
    if n_accum > 0:
        sst_clim = sst_accum / n_accum
        sss_clim = sss_accum / n_accum
        
        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 12))
        
        # SST Map
        im1 = axes2[0].imshow(sst_clim, origin='lower', cmap='RdBu_r', vmin=-2, vmax=30)
        axes2[0].set_title(f'SST Climatology (Last {n_accum} Months)')
        plt.colorbar(im1, ax=axes2[0], label='Deg C')
        
        # SSS Map
        im2 = axes2[1].imshow(sss_clim, origin='lower', cmap='viridis', vmin=32, vmax=38)
        axes2[1].set_title(f'SSS Climatology (Last {n_accum} Months)')
        plt.colorbar(im2, ax=axes2[1], label='PSU')
        
        plt.tight_layout()
        plt.savefig('prod_v3_fixed_15y_climatology.png')
        print("Saved prod_v3_fixed_15y_climatology.png")

if __name__ == "__main__":
    analyze_15y()
