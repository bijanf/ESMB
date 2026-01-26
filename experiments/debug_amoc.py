
import xarray as xr
import numpy as np

ds = xr.open_dataset("outputs/century_run/year_022.nc", decode_times=False)

if 'z' in ds:
    print("Z coords:", ds['z'].values)
else:
    print("Z coord missing!")

if 'dzt' in ds:
    print("DZT exists")
else:
    print("DZT missing")

v = ds['ocean_v'].fillna(0.0).values
print(f"V stats: Min={v.min():.4f}, Max={v.max():.4f}, Mean={v.mean():.4f}")
print(f"V abs mean: {np.abs(v).mean():.4f}")

# Check Zonal Sum at mid-lat
mid_j = int(v.shape[1] / 2)
v_slice = v[:, mid_j, :]
print(f"Mid-Lat ({mid_j}) Slice Shape: {v_slice.shape}")
print(f"Slice Sum (Zonal): {v_slice.sum(axis=1)}") 
