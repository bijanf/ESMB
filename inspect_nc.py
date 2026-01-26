import xarray as xr
ds = xr.open_dataset("outputs/production_control/mean_0436.nc", decode_times=False)
print("Dims:", ds.dims)
print("Data Vars:", list(ds.data_vars))
print("Coords:", list(ds.coords))
if 'ocean_v' in ds:
    print("ocean_v dims:", ds.ocean_v.dims)
