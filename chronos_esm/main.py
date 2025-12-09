"""
Main entry point for Chronos-ESM.

Integrates Ocean, Atmosphere, Sea Ice, and Coupler into a single differentiable model.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, Optional
from functools import partial

from chronos_esm.config import ATMOS_GRID, OCEAN_GRID, DT_ATMOS, DT_OCEAN
from chronos_esm.ocean import veros_driver as ocean_driver
from chronos_esm.ocean import diagnostics as ocean_diagnostics
from chronos_esm.atmos import dynamics as atmos_driver
from chronos_esm.atmos import physics as atmos_physics
from chronos_esm.ice import driver as ice_driver
from chronos_esm.land import driver as land_driver
from chronos_esm.land import vegetation
from chronos_esm.coupler import state as coupled_state
from chronos_esm.coupler import regrid

class ModelParams(NamedTuple):
    """Parameters for the model run."""
    co2_ppm: float = 280.0
    solar_constant: float = 1361.0
    mask: Optional[jnp.ndarray] = None # Land mask (True=Ocean)

def init_model(
    nz: int = 15,
    ny_atmos: int = ATMOS_GRID.nlat,
    nx_atmos: int = ATMOS_GRID.nlon,
    ny_ocean: int = OCEAN_GRID.nlat,
    nx_ocean: int = OCEAN_GRID.nlon
) -> coupled_state.CoupledState:
    """Initialize the full coupled model state."""
    
    # Initialize components
    ocean = ocean_driver.init_ocean_state(nz, ny_ocean, nx_ocean)
    atmos = atmos_driver.init_atmos_state(ny_atmos, nx_atmos)
    land = land_driver.init_land_state(ny_atmos, nx_atmos)
    
    # Initialize coupled state (creates Ice and Fluxes internally)
    state = coupled_state.init_coupled_state(ocean, atmos, land)
    
    return state


@partial(jax.jit, static_argnames=['regridder'])
def step_coupled(
    state: coupled_state.CoupledState,
    params: ModelParams,
    regridder: regrid.Regridder
) -> coupled_state.CoupledState:
    """
    Perform one coupled time step.
    """
    
    # 1. Atmosphere Step
    beta = jnp.ones_like(state.fluxes.sst)
    
    if hasattr(params, 'mask') and params.mask is not None:
        ocean_mask = params.mask # 1=Ocean, 0=Land
        land_mask = 1.0 - ocean_mask
        
        # Land Beta
        if hasattr(state, 'land') and state.land is not None:
            BUCKET_DEPTH = 0.15 
            _, veg_fraction = vegetation.compute_land_properties(state.land.lai)
            bucket_beta = state.land.soil_moisture / BUCKET_DEPTH
            bucket_beta = jnp.clip(bucket_beta, 0.0, 1.0)
            land_beta = bucket_beta * (1.0 + veg_fraction)
            land_beta = jnp.clip(land_beta, 0.0, 1.0)
            beta = ocean_mask * 1.0 + land_mask * land_beta
            
    # Carbon Cycle Coupling
    dic_surf = regridder.ocean_to_atmos(state.ocean.dic[0])
    pco2_sea = dic_surf * (280.0 / 2000.0)
    pco2_air = state.atmos.co2
    
    k_gas = 1.0e-8 
    flux_c_sea = k_gas * (pco2_sea - pco2_air) # Positive Upward (Sea->Air)
    
    # Air-Land Flux (Lagged)
    flux_c_land = state.fluxes.carbon_flux_land
    
    # Total Flux to Atmosphere [ppm/s]
    co2_conv_atm = 240.0
    
    if hasattr(params, 'mask') and params.mask is not None:
        ocean_mask = params.mask
        land_mask = 1.0 - ocean_mask
        flux_c_total = ocean_mask * flux_c_sea + land_mask * flux_c_land
    else:
        flux_c_total = flux_c_sea
        
    co2_flux_atm_ppm = flux_c_total * co2_conv_atm
    
    # Calculate Surface Fluxes for Atmos
    sensible_flux, latent_flux = atmos_physics.compute_surface_fluxes(
        temp_air=state.atmos.temp,
        q_air=state.atmos.q,
        u_air=state.atmos.u,
        v_air=state.atmos.v,
        temp_surf=state.fluxes.sst, # Composite SST
        beta=beta
    )
    
    new_atmos, (precip_atm, sfc_pressure) = atmos_driver.step_atmos(
        state.atmos,
        surface_temp=state.fluxes.sst,
        flux_sensible=sensible_flux,
        flux_latent=latent_flux,
        flux_co2=co2_flux_atm_ppm,
        dt=DT_ATMOS,
        ny=ATMOS_GRID.nlat,
        nx=ATMOS_GRID.nlon
    )
    
    # Net Heat Flux to Ocean (Positive Down)
    sw_net = 240.0 # W/m2 (Positive Down)
    lw_net = 50.0  # W/m2 (Net Upward) -> Cooling
    net_heat_atm = sw_net - lw_net - sensible_flux - latent_flux
    
    # Freshwater Flux (Positive = P - E)
    evap = latent_flux / 2.5e6
    fw_flux_atm = precip_atm - evap
    
    # 2. Coupler: Atmos -> Ocean/Ice
    heat_flux_ocn = regridder.atmos_to_ocean(net_heat_atm)
    fw_flux_ocn = regridder.atmos_to_ocean(fw_flux_atm)
    
    # Zonal Wind Stress (Sinusoidal to drive Gyres)
    # tau_x = 0.1 * cos(3 * lat) ?
    # Let's use a simple profile: Westerlies in mid-lat, Easterlies in tropics
    lat = jnp.linspace(-90, 90, ATMOS_GRID.nlat)
    lat_rad = jnp.deg2rad(lat)
    tau_profile = 0.1 * jnp.sin(6.0 * lat_rad) # Multiple bands
    tau_x_atm = jnp.broadcast_to(tau_profile[:, None], net_heat_atm.shape)
    tau_y_atm = jnp.zeros_like(net_heat_atm)
    
    tau_x_ocn = regridder.atmos_to_ocean(tau_x_atm)
    tau_y_ocn = regridder.atmos_to_ocean(tau_y_atm)
    
    # 3. Sea Ice Step
    t_air_ocn = regridder.atmos_to_ocean(new_atmos.temp - 273.15) # K -> C
    sw_down = jnp.zeros_like(t_air_ocn) 
    lw_down = jnp.ones_like(t_air_ocn) * 300.0
    sst_ocean_grid = state.ocean.temp[0] - 273.15 # Top layer, K -> C
    
    new_ice, (ice_heat_flux, ice_fw_flux) = ice_driver.step_ice(
        state.ice,
        t_air=t_air_ocn,
        sw_down=sw_down,
        lw_down=lw_down,
        ocean_temp=sst_ocean_grid,
        ny=OCEAN_GRID.nlat,
        nx=OCEAN_GRID.nlon,
        mask=params.mask
    )
    
    # 4. Ocean Step
    A = new_ice.concentration
    combined_heat_flux = (1.0 - A) * heat_flux_ocn + A * ice_heat_flux
    combined_fw_flux = (1.0 - A) * fw_flux_ocn + A * ice_fw_flux
    
    fluxes_ocean = (combined_heat_flux, combined_fw_flux, -flux_c_sea)
    wind_ocean = (tau_x_ocn, tau_y_ocn)
    
    dx_ocn = 100e3
    dy_ocn = 100e3
    dz_ocn = jnp.ones(state.ocean.u.shape[0]) * 100.0
    
    new_ocean = ocean_driver.step_ocean(
        state.ocean,
        surface_fluxes=fluxes_ocean,
        wind_stress=wind_ocean,
        dx=dx_ocn,
        dy=dy_ocn,
        dz=dz_ocn,
        nz=state.ocean.u.shape[0],
        ny=OCEAN_GRID.nlat,
        nx=OCEAN_GRID.nlon,
        mask=params.mask,
        dt=DT_ATMOS
    )
    
    # Apply Bathymetry Mask
    if hasattr(params, 'mask') and params.mask is not None:
        mask = params.mask
        mask_3d = jnp.broadcast_to(mask, new_ocean.u.shape)
        u_masked = jnp.where(mask_3d, new_ocean.u, 0.0)
        v_masked = jnp.where(mask_3d, new_ocean.v, 0.0)
        new_ocean = new_ocean._replace(u=u_masked, v=v_masked)
        
    # 5. Land Step
    if hasattr(params, 'mask') and params.mask is not None:
        ocean_mask = params.mask
        land_mask = 1.0 - ocean_mask
        
        sw_down = jnp.ones_like(new_atmos.temp) * 240.0
        lw_down = jnp.ones_like(new_atmos.temp) * 300.0
        precip_proxy = jnp.maximum(fw_flux_atm, 0.0) / 1000.0 # kg/m2/s -> m/s
        
        new_land, (land_sensible, land_latent, land_nee) = land_driver.step_land(
            state.land,
            t_air=new_atmos.temp,
            q_air=new_atmos.q,
            sw_down=sw_down,
            lw_down=lw_down,
            precip=precip_proxy,
            mask=land_mask
        )
    else:
        new_land = state.land
        land_mask = jnp.zeros_like(new_atmos.temp)
        land_sensible = jnp.zeros_like(new_atmos.temp)
        land_latent = jnp.zeros_like(new_atmos.temp)
        land_nee = jnp.zeros_like(new_atmos.temp)

    # 6. Coupler: Surface -> Atmos
    sst_ocean_ice = (1.0 - A) * (new_ocean.temp[0]) + A * (new_ice.surface_temp + 273.15)
    
    if hasattr(params, 'mask') and params.mask is not None:
        ocean_mask = params.mask
        sst_composite = ocean_mask * sst_ocean_ice + (1.0 - ocean_mask) * new_land.temp
    else:
        sst_composite = sst_ocean_ice
    
    sst_atm = regridder.ocean_to_atmos(sst_composite)
    
    new_fluxes = coupled_state.FluxState(
        net_heat_flux=net_heat_atm,
        freshwater_flux=fw_flux_atm,
        wind_stress_x=tau_x_atm,
        wind_stress_y=tau_y_atm,
        precip=precip_atm,
        sst=sst_atm,
        carbon_flux_ocean=flux_c_sea,
        carbon_flux_land=land_nee
    )
    
    return coupled_state.CoupledState(
        ocean=new_ocean,
        atmos=new_atmos,
        ice=new_ice,
        land=new_land,
        fluxes=new_fluxes,
        time=state.time + DT_ATMOS
    )


def run_simulation(
    steps: int,
    params: ModelParams = ModelParams()
) -> coupled_state.CoupledState:
    """
    Run the coupled simulation.
    """
    state = init_model()
    regridder = regrid.Regridder()
    
    def scan_fn(carry, _):
        state = carry
        new_state = step_coupled(state, params, regridder)
        return new_state, None
        
    final_state, _ = jax.lax.scan(scan_fn, state, jnp.arange(steps))
    
    return final_state


if __name__ == "__main__":
    import time
    print("Initializing Chronos-ESM...")
    # Run a short warm-up
    print("Running warm-up (10 steps)...")
    state = run_simulation(10)
    
    print("Running simulation (100 steps)...")
    t0 = time.time()
    
    # Create a mask for testing
    # mask = jnp.zeros((96, 192))
    # mask = mask.at[:, :96].set(1.0) # Left half Ocean
    # params = ModelParams(mask=mask)
    
    final_state = run_simulation(100)
    # Force synchronization
    final_state.ocean.temp.block_until_ready()
    t1 = time.time()
    
    print(f"Simulation complete in {t1-t0:.2f}s")
    print(f"Final Global Mean Temp: {jnp.mean(final_state.atmos.temp):.2f} K")
    
    # Compute AMOC
    amoc = ocean_diagnostics.compute_amoc_index(final_state.ocean)
    print(f"Final AMOC Index: {amoc:.2f} Sv")
