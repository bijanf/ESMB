
import xarray as xr
import numpy as np

def find_hotspots(filename):
    print(f"Analyzing {filename}...")
    ds = xr.open_dataset(filename, decode_times=False)
    
    # SST
    sst = ds.ocean_temp.isel(z=0).values - 273.15
    mask = ds.mask.values if 'mask' in ds else np.ones_like(sst)
    
    # Mask Land (assuming 0.0 is land from previous fix)
    # But wait, 0.0 C is -273.15 K?
    # No, model temp is Kelvin. Clean land is 0.0 K.
    # So sst (vals - 273.15) on land would be -273.15 C.
    
    # Let's filter for physical range
    valid_sst = np.where(ds.ocean_temp.isel(z=0).values > 1.0, sst, np.nan)
    
    max_val = np.nanmax(valid_sst)
    min_val = np.nanmin(valid_sst)
    
    print(f"Global Max SST: {max_val:.2f} C")
    print(f"Global Min SST: {min_val:.2f} C")
    
    # Find Location of Max
    max_idx = np.unravel_index(np.nanargmax(valid_sst), valid_sst.shape)
    # y = lat index, x = lon index
    y_idx, x_idx = max_idx
    
    # Get Lat/Lon coords if available
    if 'y_ocn' in ds.coords:
        lat = ds.y_ocn[y_idx].values
        lon = ds.x_ocn[x_idx].values
    else:
        # manual freq
        lat = -90 + y_idx * (180/96)
        lon = x_idx * (360/192)
        
    print(f"Hotspot Location: Index ({y_idx}, {x_idx}), Lat: {lat:.1f}, Lon: {lon:.1f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    args = parser.parse_args()
    find_hotspots(args.filename)
