
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cartopy.crs as ccrs

def plot_climate(nc_file):
    print(f"Plotting {nc_file}...")
    ds = xr.open_dataset(nc_file, decode_times=False)
    
    # Create output dir
    out_dir = Path(nc_file).parent / "plots"
    out_dir.mkdir(exist_ok=True)
    
    # Assign Coordinates (Indices -> Degs)
    nx = ds.sizes['x_atm']
    ny = ds.sizes['y_atm']
    lon = np.linspace(0, 360, nx, endpoint=False)
    lat = np.linspace(-90, 90, ny)
    
    ds = ds.assign_coords(x_atm=lon, y_atm=lat)
    ds = ds.rename({'x_atm': 'lon', 'y_atm': 'lat'})
    
    # 1. Precipitation Map
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    
    # Precip is in kg/m2/s in file??
    # Wait, io.py saves average precip.
    # Unit check: if it's kg/m2/s, mult by 86400 -> mm/day.
    
    precip = ds['precip'] * 86400.0
    precip.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='Blues', vmin=0, vmax=10, cbar_kwargs={'label': 'mm/day'})
    ax.set_title(f"Precipitation Rate (Global Mean: {precip.mean():.2f} mm/day)")
    plt.savefig(out_dir / "precip_map.png")
    plt.close()
    
    # 2. Temperature Map (Atmos Surface)
    # Check if we have sst or atmos_temp
    # atmos_temp is air temp.
    plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    
    temp = ds['atmos_temp']
    temp.plot(ax=ax, transform=ccrs.PlateCarree(), cmap='RdBu_r')
    ax.set_title("Atmospheric Temperature [K]")
    plt.savefig(out_dir / "temp_map.png")
    plt.close()

    # 3. Zonal Mean Precip
    plt.figure(figsize=(8, 5))
    precip_zonal = precip.mean(dim='lon')
    lat = ds.lat.values
    plt.plot(lat, precip_zonal)
    plt.xlabel('Latitude')
    plt.ylabel('Precipitation (mm/day)')
    plt.title('Zonal Mean Precipitation')
    plt.grid(True)
    # Add target ITCZ reference
    target = 8.0 * np.exp(-(lat/10.0)**2) + 3.0 * np.exp(-( (np.abs(lat)-45)/15 )**2)
    plt.plot(lat, target, 'r--', alpha=0.5, label='Target Profile')
    plt.legend()
    plt.savefig(out_dir / "zonal_precip.png")
    plt.close()
    
    print(f"Plots saved to {out_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        plot_climate(sys.argv[1])
    else:
        # Default
        plot_climate("outputs/verify_precip/verify_state.nc")
