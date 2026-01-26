
import xarray as xr
import numpy as np

def check_poles():
    ds = xr.open_dataset("outputs/long_run_fast/final_state.nc", decode_times=False)
    
    # Atmos Temp
    temp = ds['atmos_temp'].values
    ny, nx = temp.shape
    
    # Assuming lat is linspace(-90, 90)
    # Index 0 = South Pole
    # Index -1 = North Pole
    
    sp_temp = np.mean(temp[0, :])
    np_temp = np.mean(temp[-1, :])
    
    print(f"South Pole (Index 0) Mean Temp: {sp_temp:.2f} K ({sp_temp-273.15:.2f} C)")
    print(f"North Pole (Index -1) Mean Temp: {np_temp:.2f} K ({np_temp-273.15:.2f} C)")
    
    # Check Time/Season
    time_sec = ds['time'].values
    day = (float(time_sec) % (365*86400)) / 86400.0
    print(f"Day of Year: {day:.1f} (0=Jan 1)")
    
    # Check if we have Solar Insolation DIAGNOSTIC saved?
    # io.py does NOT save sw_down.
    # We might need to re-calculate it to verify.
    
    # Check Latent/Sensible Fluxes if available (fluxes struct)
    # They are averaged.
    
    ds.close()

if __name__ == "__main__":
    check_poles()
