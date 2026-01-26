import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import glob

def analyze_control_run():
    output_dir = Path("outputs/production_control")
    analysis_dir = Path("analysis_results/control_run_36y")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    # Load all monthly means
    files = sorted(list(output_dir.glob("mean_*.nc")))
    # Limit to first 436 if more exist now (but we only saw 436)
    
    if not files:
        print("No files found!")
        return

    print("Processing files sequentially (no dask)...")
    
    gmt_series = []
    amoc_series = []
    snow_series = []
    
    # Pre-calculate weights from first file
    with xr.open_dataset(files[0], decode_times=False) as ds_first:
        if 'y_atm' in ds_first.coords:
            lats = ds_first.y_atm
            weights = np.cos(np.deg2rad(lats))
            weights.name = "weights"
        else:
            weights = None
            
    for i, f in enumerate(files):
        if i % 12 == 0:
            print(f"Processing year {i//12 + 1}...")
            
        with xr.open_dataset(f, decode_times=False) as ds:
            # 1. GMT
            if weights is not None:
                gmt_val = ds.atmos_temp.weighted(weights).mean(dim=["y_atm", "x_atm"]).values
            else:
                gmt_val = ds.atmos_temp.mean().values
            gmt_series.append(gmt_val)
            
            # 2. AMOC Proxy
            # Sum v along lon (x_ocn)
            moc_proxy = ds.ocean_v.sum(dim='x_ocn') # (z, y_ocn)
            # Cumsum along z (assuming z=0 surface to bottom, or vice versa, either way max abs val works for strength)
            moc_approx = moc_proxy.cumsum(dim='z')
            amoc_val = np.max(moc_approx.values)
            amoc_series.append(amoc_val)
            
            # 3. Snow
            snow_val = ds.land_snow_depth.sum().values
            snow_series.append(snow_val)

    # Plotting
    months = np.arange(1, len(files) + 1)
    
    # GMT
    plt.figure(figsize=(10, 5))
    plt.plot(months, gmt_series)
    plt.title("Global Mean Surface Air Temperature (Control Run)")
    plt.ylabel("Temperature (K)")
    plt.xlabel("Month")
    plt.grid(True)
    plt.savefig(analysis_dir / "gmt_timeseries.png")
    plt.close()

    # AMOC
    plt.figure(figsize=(10, 5))
    plt.plot(months, amoc_series)
    plt.title("Global Overturning Index (Qualitative Unit)")
    plt.ylabel("Index (Transport Proxy)")
    plt.xlabel("Month")
    plt.grid(True)
    plt.savefig(analysis_dir / "amoc_proxy_timeseries.png")
    plt.close()

    # Snow
    plt.figure(figsize=(10, 5))
    plt.plot(months, snow_series)
    plt.title("Total Land Snow Volume (Proxy)")
    plt.ylabel("Volume Index")
    plt.xlabel("Month")
    plt.grid(True)
    plt.savefig(analysis_dir / "land_snow_timeseries.png")
    plt.close()

    # 4. Spatial Plots (Last Month)
    print("Generating spatial plots for last month...")
    with xr.open_dataset(files[-1], decode_times=False) as last_step:
        # SST
        plt.figure(figsize=(12, 6))
        last_step.sst.plot(cmap='RdBu_r')
        plt.title(f"SST (K) - Month {len(files)}")
        plt.savefig(analysis_dir / "sst_map_final.png")
        plt.close()

        # Surface Air Temp
        plt.figure(figsize=(12, 6))
        last_step.atmos_temp.plot(cmap='RdBu_r', vmin=220, vmax=320)
        plt.title(f"Surface Air Temp (K) - Month {len(files)}")
        plt.savefig(analysis_dir / "sat_map_final.png")
        plt.close()
        
        # Ocean Temp Zonal Mean
        plt.figure(figsize=(10, 6))
        last_step.ocean_temp.mean(dim='x_ocn').plot(yincrease=False)
        plt.title(f"Ocean Temp Zonal Mean (K) - Month {len(files)}")
        plt.savefig(analysis_dir / "ocean_temp_zonal_final.png")
        plt.close()

    print(f"Analysis complete. Results in {analysis_dir}")

if __name__ == "__main__":
    analyze_control_run()
