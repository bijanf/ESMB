"""
Resume Control Run.

Loads the latest restart_XXXX.nc file and continues the simulation, 
matching the physics and output format of control_run.py.
"""

import glob
import sys
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import xarray as xr

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import data
from chronos_esm import main 
from chronos_esm import io as model_io
from chronos_esm import log_utils as c_log
from chronos_esm.ocean import veros_driver
from chronos_esm.atmos import dynamics as atmos_driver
from chronos_esm.coupler import state as coupled_state
from chronos_esm.land import driver as land_driver
from chronos_esm.ice import driver as ice_driver

def load_latest_state(output_dir: Path):
    """Load the latest NetCDF checkpoint (restart_*.nc)."""
    # Prioritize 'restart_*.nc' files as they are full snapshots
    files = sorted(glob.glob(str(output_dir / "restart_*.nc")))
    
    if not files:
        # Fallback to state_*.nc if needed, but control_run uses restart_*.nc for checkpoints
        files = sorted(glob.glob(str(output_dir / "state_*.nc")))
        
    if not files:
        raise FileNotFoundError(f"No restart files found in {output_dir}")

    latest_file = files[-1]
    print(f"Loading state from {latest_file}...")

    ds = xr.open_dataset(latest_file, decode_times=False)

    # Reconstruct State (similar to control_run init but from file)
    
    # Ocean
    ocean = veros_driver.OceanState(
        u=jnp.array(ds.ocean_u.values),
        v=jnp.array(ds.ocean_v.values),
        w=jnp.array(ds.ocean_w.values) if 'ocean_w' in ds else jnp.zeros_like(jnp.array(ds.ocean_u.values)),
        temp=jnp.array(ds.ocean_temp.values),
        salt=jnp.array(ds.ocean_salt.values),
        psi=jnp.array(ds.ocean_psi.values) if 'ocean_psi' in ds else jnp.zeros((ds.ocean_u.shape[1], ds.ocean_u.shape[2])),
        rho=jnp.zeros_like(jnp.array(ds.ocean_temp.values)), # Will be updated
        dic=jnp.array(ds.ocean_dic.values) if 'ocean_dic' in ds else jnp.ones_like(jnp.array(ds.ocean_temp.values)) * 2000.0,
    )

    # Atmos
    # Load topography for phi_s
    ny, nx = ds.atmos_temp.shape
    try:
        topo_m = data.load_topography(ny, nx)
        phi_s = topo_m * 9.81
    except:
        phi_s = jnp.zeros((ny, nx))

    atmos = atmos_driver.AtmosState(
        u=jnp.array(ds.atmos_u.values),
        v=jnp.array(ds.atmos_v.values),
        temp=jnp.array(ds.atmos_temp.values),
        q=jnp.array(ds.atmos_q.values),
        co2=jnp.array(ds.atmos_co2.values) if 'atmos_co2' in ds else jnp.ones_like(jnp.array(ds.atmos_temp.values)) * 280.0,
        vorticity=jnp.array(ds.atmos_vorticity.values) if 'atmos_vorticity' in ds else jnp.zeros_like(jnp.array(ds.atmos_temp.values)),
        divergence=jnp.array(ds.atmos_divergence.values) if 'atmos_divergence' in ds else jnp.zeros_like(jnp.array(ds.atmos_temp.values)),
        ln_ps=jnp.array(ds.atmos_ln_ps.values) if 'atmos_ln_ps' in ds else jnp.zeros_like(jnp.array(ds.atmos_temp.values)),
        psi=jnp.zeros((ny, nx)), # Diagnostic
        chi=jnp.zeros((ny, nx)), # Diagnostic
        phi_s=phi_s
    )

    # Ice (if present in restart, otherwise init)
    # Control run might not save ice explicitly in restart if not added to IO? 
    # Checking control_run.py... it just saves "state". 
    # Does io.save_state_to_netcdf save ice? Assuming yes.
    
    # If not in DS, init empty/default
    if 'ice_thickness' in ds:
         ice = ice_driver.IceState(
            thickness=jnp.array(ds.ice_thickness.values),
            concentration=jnp.array(ds.ice_concentration.values),
            surface_temp=jnp.array(ds.ice_surface_temp.values) if 'ice_surface_temp' in ds else jnp.ones_like(jnp.array(ds.ice_thickness.values)) * -1.8
         )
    else:
         # Init default ice
         print("Warning: Ice not found in restart, valid if no ice model enabled. Initializing empty.")
         ice = ice_driver.init_ice_state(ds.ocean_u.shape[1], ds.ocean_u.shape[2])


    # Land
    # Re-init or load
    if 'land_temp' in ds:
         land = land_driver.LandState(
             temp=jnp.array(ds.land_temp.values),
             soil_moisture=jnp.array(ds.land_soil_moisture.values) if 'land_soil_moisture' in ds else jnp.ones_like(jnp.array(ds.land_temp.values)) * 0.5,
             snow_depth=jnp.array(ds.land_snow_depth.values) if 'land_snow_depth' in ds else jnp.zeros_like(jnp.array(ds.land_temp.values)),
             lai=jnp.array(ds.land_lai.values) if 'land_lai' in ds else jnp.ones_like(jnp.array(ds.land_temp.values)) * 1.0,
             soil_carbon=jnp.array(ds.land_soil_carbon.values) if 'land_soil_carbon' in ds else jnp.ones_like(jnp.array(ds.land_temp.values)) * 10.0,
         )
    else:
         print("Warning: Land not found in restart. Re-initializing.")
         land = land_driver.init_land_state(ds.ocean_u.shape[1], ds.ocean_u.shape[2])

    # Fluxes
    fluxes = coupled_state.FluxState(
        net_heat_flux=jnp.array(ds.net_heat_flux.values) if 'net_heat_flux' in ds else jnp.zeros_like(atmos.temp),
        freshwater_flux=jnp.array(ds.freshwater_flux.values) if 'freshwater_flux' in ds else jnp.zeros_like(atmos.temp),
        wind_stress_x=jnp.zeros_like(atmos.temp), # Diagnostic
        wind_stress_y=jnp.zeros_like(atmos.temp),
        sst=jnp.array(ds.sst.values) if 'sst' in ds else jnp.array(ds.ocean_temp.values)[0], # Use Ocean Surface if SST missing
        carbon_flux_ocean=jnp.zeros_like(atmos.temp),
        carbon_flux_land=jnp.zeros_like(atmos.temp),
        precip=jnp.array(ds.precip.values) if 'precip' in ds else jnp.zeros_like(atmos.temp), # Required
    )

    # Time
    time_val = float(ds.time.values)

    state = coupled_state.CoupledState(
        ocean=ocean, atmos=atmos, ice=ice, land=land, fluxes=fluxes, time=time_val
    )

    # Extract month index from filename (restart_XXXX.nc)
    try:
        month_idx = int(Path(latest_file).stem.split("_")[1])
    except:
        month_idx = 0
        print(f"Could not parse month from {latest_file}, assuming 0.")

    return state, month_idx


def resume_run(target_years: float = 100.0):
    """Resume simulation."""
    output_dir = Path("outputs/control_run")

    # Load State
    try:
        current_state, start_month = load_latest_state(output_dir)
    except Exception as e:
        print(f"Failed to load state: {e}")
        return

    # Setup Logger (append mode?)
    logger = c_log.setup_logger("resume_run", output_dir / f"resume_{start_month}.log")
    logger.info(f"Resuming run from Month {start_month}...")
    logger.info(f"Loaded Time: {current_state.time/86400/365:.2f} years")

    # Config
    seconds_per_year = 365.0 * 24.0 * 3600.0
    seconds_per_month = seconds_per_year / 12.0
    
    # Use consistent physics timestep
    from chronos_esm.config import DT_OCEAN 
    dt = DT_OCEAN
    
    steps_per_month = int(seconds_per_month / dt)
    n_months = int(target_years * 12)

    regridder = main.regrid.Regridder()
    mask = data.load_bathymetry_mask()
    # Use consistent Params (e.g. constant CO2)
    params = main.ModelParams(mask=mask, co2_increase_rate=0.0)

    # ------------------------------------------------------------------------
    # Compile Step (Matching control_run.py loop structure with accumulation)
    # ------------------------------------------------------------------------
    
    # We need the Accumulator class to be compatible with JAX Pytree
    # We can use the State objects themselves as accumulators or explicit tuple
    
    def zero_accumulator(s):
        # Create a structure of zeros matching the state, but we only need fields we mean
        # For simplicity, let's use the explicit tuple approach from control_run
        # Or better: accumulate the wHOLE state? No, that's heavy.
        # Let's clone control_run logic exactly.
        
        # We need to replicate the 'Accumulator' NamedTuple or just use a dict/tuple
        # Tuple is easiest for JIT
        
        # Order: ocean_temp, ocean_salt, ocean_u, ocean_v, ocean_w, ocean_psi, ocean_dic,
        #        atmos_temp, atmos_u, atmos_v, atmos_q, atmos_co2, 
        #        sst, precip
        # ... this is tedious to keep synced. 
        # Let's accumulate 'current_state' directly? No, that mixes types.
        
        # Let's use a simple list of leaf nodes to accumulate
        leaves, treedef = jax.tree_util.tree_flatten(s)
        zero_leaves = [jnp.zeros_like(l) for l in leaves]
        return zero_leaves, treedef

    def month_step_fn(carry, _):
        s, acc_leaves = carry
        # Use Tuned Physics Parameter r_drag=7.9e-4
        new_s = main.step_coupled(s, params, regridder, r_drag=7.9e-4) 

        # Accumulate
        current_leaves = jax.tree_util.tree_leaves(new_s)
        new_acc_leaves = [a + c for a, c in zip(acc_leaves, current_leaves)]

        return (new_s, new_acc_leaves), None

    @jax.jit
    def run_one_month(start_state):
        # Init acc
        init_acc_leaves, treedef = zero_accumulator(start_state)
        
        # Scan
        (final_s, final_acc_leaves), _ = jax.lax.scan(
            month_step_fn, (start_state, init_acc_leaves), jnp.arange(steps_per_month)
        )
        
        # Reconstruct mean state
        mean_leaves = [l / steps_per_month for l in final_acc_leaves]
        mean_state = jax.tree_util.tree_unflatten(treedef, mean_leaves)
        
        return final_s, mean_state

    # ------------------------------------------------------------------------
    
    logger.info("Starting simulation loop...")

    # Continue from next month
    for month in range(start_month + 1, n_months + 1):
        logger.info(f"Running Month {month}/{n_months}...")
        t_start = time.time()

        current_state, mean_state = run_one_month(current_state)
        
        # Block
        current_state.ocean.temp.block_until_ready()
        t_end = time.time()

        # Stability Check
        t_mean = float(jnp.mean(current_state.atmos.temp))
        if jnp.isnan(t_mean) or t_mean > 350.0 or t_mean < 150.0:
            logger.error(f"Stability Check Failed! Global Mean Temp: {t_mean} K")
            break

        logger.info(
            f"Month {month} complete in {t_end - t_start:.2f}s. "
            f"T_mean={t_mean:.2f}K."
        )
        
        # Save Monthly Mean matching control_run naming
        model_io.save_state_to_netcdf(
            mean_state, output_dir / f"mean_{month:04d}.nc"
        )
        
        # Save Restart every 12 months
        if month % 12 == 0:
            logger.info(f"Saving Restart Checkpoint at Month {month}...")
            model_io.save_state_to_netcdf(current_state, output_dir / f"restart_{month:04d}.nc")


if __name__ == "__main__":
    resume_run(target_years=100.0)
