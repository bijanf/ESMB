
import xarray as xr
import numpy as np

def locate_max_wind(path):
    ds = xr.open_dataset(path, decode_times=False)
    u = ds['atmos_u'].values
    v = ds['atmos_v'].values
    speed = np.sqrt(u**2 + v**2)
    
    max_idx = np.unravel_index(np.argmax(speed), speed.shape)
    print(f"Max Wind: {np.max(speed):.2f} m/s")
    print(f"Location (y, x): {max_idx}")
    
    print("Slice around max:")
    y, x = max_idx
    if y > 2 and y < speed.shape[0]-2:
        print(speed[y-2:y+3, max(0, x-2):min(speed.shape[1], x+3)])

if __name__ == "__main__":
    locate_max_wind("outputs/century_run/year_038.nc")
