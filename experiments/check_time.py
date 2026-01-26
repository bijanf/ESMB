
import xarray as xr
import sys
import os

start = 390
end = 401
run_dir = "outputs/control_run_prod_v4"

for m in range(start, end+1):
    f = os.path.join(run_dir, f"mean_{m:04d}.nc")
    if os.path.exists(f):
        ds = xr.open_dataset(f, decode_times=False)
        t = ds.time.values
        # scalar or array?
        t_val = float(t) if t.shape == () else float(t[0])
        
        seconds_in_year = 365.0 * 86400.0
        years = t_val / seconds_in_year
        day = (t_val % seconds_in_year) / 86400.0
        
        print(f"Month {m}: Time={t_val:.1f} s = {years:.2f} Years. Day of Year={day:.2f}")
    else:
        print(f"Month {m}: Missing")
