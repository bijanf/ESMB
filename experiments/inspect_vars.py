
import xarray as xr
ds = xr.open_dataset("outputs/century_run/year_022.nc", decode_times=False)
print(ds.data_vars)
ds.close()
