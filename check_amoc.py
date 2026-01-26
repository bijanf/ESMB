
import xarray as xr
import jax.numpy as jnp
from chronos_esm.ocean import diagnostics
from chronos_esm.ocean import veros_driver

def check_amoc():
    path = "outputs/control_run/mean_0023.nc" 
    print(f"Loading {path}...")
    ds = xr.open_dataset(path, decode_times=False)
    
    # Reconstruct OceanState (Minimal for MOC)
    u_val = jnp.array(ds.ocean_u.values)
    ocean = veros_driver.OceanState(
        u=u_val,
        v=jnp.array(ds.ocean_v.values),
        w=jnp.zeros_like(u_val),
        temp=jnp.array(ds.ocean_temp.values),
        salt=jnp.array(ds.ocean_salt.values),
        psi=jnp.zeros((ds.ocean_u.shape[1], ds.ocean_u.shape[2])),
        rho=jnp.zeros_like(jnp.array(ds.ocean_temp.values)),
        dic=jnp.zeros_like(jnp.array(ds.ocean_temp.values)),
    )
    
    amoc = diagnostics.compute_amoc_index(ocean)
    print(f"AMOC Index (Month 23): {amoc:.2f} Sv")

if __name__ == "__main__":
    check_amoc()
