"""
Main entry point for Chronos-ESM.

Integrates Ocean, Atmosphere, Sea Ice, and Coupler into a single differentiable model.
"""

from functools import partial
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from chronos_esm.atmos import dynamics as atmos_driver
from chronos_esm.atmos import physics as atmos_physics
from chronos_esm.config import ATMOS_GRID, DT_ATMOS, DT_OCEAN, OCEAN_GRID
from chronos_esm.coupler import regrid
from chronos_esm.coupler import state as coupled_state
from chronos_esm.ice import driver as ice_driver
from chronos_esm.land import driver as land_driver
from chronos_esm.land import vegetation
from chronos_esm.ocean import diagnostics as ocean_diagnostics
from chronos_esm.ocean import veros_driver as ocean_driver


class ModelParams(NamedTuple):
    """Parameters for the model run."""

    co2_ppm: float = 280.0
    co2_increase_rate: float = 0.0  # Fractional increase per year (e.g. 0.01 for 1%)
    solar_constant: float = 1361.0
    mask: Optional[jnp.ndarray] = None  # Land mask (True=Ocean)


def init_model(
    nz: int = 15,
    ny_atmos: int = ATMOS_GRID.nlat,
    nx_atmos: int = ATMOS_GRID.nlon,
    ny_ocean: int = OCEAN_GRID.nlat,
    nx_ocean: int = OCEAN_GRID.nlon,
) -> coupled_state.CoupledState:
    """Initialize the full coupled model state."""

    # Initialize components
    ocean = ocean_driver.init_ocean_state(nz, ny_ocean, nx_ocean)
    atmos = atmos_driver.init_atmos_state(ny_atmos, nx_atmos)
    land = land_driver.init_land_state(ny_atmos, nx_atmos)

    # Initialize coupled state (creates Ice and Fluxes internally)
    state = coupled_state.init_coupled_state(ocean, atmos, land)

    return state


@partial(jax.jit, static_argnames=["regridder"])
def step_coupled(
    state: coupled_state.CoupledState, params: ModelParams, regridder: regrid.Regridder, 
    r_drag: float = 5.0e-2, kappa_gm: float = 1000.0, kappa_h: float = 310.0, kappa_bi: float = 3.9e14,
    Ah: float = 8.1e4, Ab: float = 0.0, shapiro_strength: float = 0.0, smag_constant: float = 0.1,
    physics_params: Optional[dict] = None
) -> coupled_state.CoupledState:
    """
    Perform one coupled time step.
    Propagates Atmosphere/Land with DT_ATMOS for N steps.
    Propagates Ocean/Ice with DT_OCEAN for 1 step.
    """
    
    # Calculate sub-steps
    # DT_OCEAN is e.g. 900s, DT_ATMOS is 30s -> 30 sub-steps
    n_substeps = int(DT_OCEAN / DT_ATMOS)

    # 1. Sub-step Loop for Atmosphere & Land
    
    def atmos_land_step(carry, _):
        # Unpack carry
        (current_state, acc_fluxes) = carry
        
        # -----------------------------------------------------
        # ATMOSPHERE & LAND DYNAMICS
        # -----------------------------------------------------
        
        # 0. Apply CO2 Forcing (Prescribed)
        seconds_per_year = 365.0 * 24.0 * 3600.0
        years_elapsed = current_state.time / seconds_per_year
        current_co2_ppm = params.co2_ppm * jnp.power(1.0 + params.co2_increase_rate, years_elapsed)
        co2_field = jnp.ones_like(current_state.atmos.co2) * current_co2_ppm
        current_state = current_state._replace(atmos=current_state.atmos._replace(co2=co2_field))

        # 1. Surface Fluxes logic (Beta calculation)
        beta = jnp.ones_like(current_state.fluxes.sst)
        ocean_mask = params.mask if (hasattr(params, "mask") and params.mask is not None) else 1.0
        # Cast to float to ensure correct arithmetic
        if hasattr(ocean_mask, 'dtype') and ocean_mask.dtype == bool:
             ocean_mask = ocean_mask.astype(jnp.float32)
        
        land_mask = 1.0 - ocean_mask
        
        
        

        if hasattr(current_state, "land") and current_state.land is not None:
             # Simple Land Beta
             BUCKET_DEPTH = 0.15
             _, veg_fraction = vegetation.compute_land_properties(current_state.land.lai)
             bucket_beta = current_state.land.soil_moisture / BUCKET_DEPTH
             bucket_beta = jnp.clip(bucket_beta, 0.0, 1.0)
             land_beta = bucket_beta * (1.0 + veg_fraction)
             land_beta = jnp.clip(land_beta, 0.0, 1.0)
             beta = ocean_mask * 1.0 + land_mask * land_beta
        
        # 1. Physics & Fluxes
        # -------------------
        
        # A. Land Fluxes (Peek)
        # We need fluxes to drive Atmos, but Land state update needs Precip from Atmos.
        # Solution: Run step_land with dummy precip to get fluxes. 
        # Fluxes depend on State(t) and T_air(t), not Precip(t) (explicitly).
        
        # Calculate Driving Winds and Drag for Land
        topo_height = current_state.atmos.phi_s / 9.81
        z0 = 0.0001 + 0.001 * topo_height
        z0 = jnp.clip(z0, 0.0001, 5.0)
        k_von_karman = 0.4
        z_lev = 50.0 
        cd = (k_von_karman / jnp.log(z_lev / z0)) ** 2
        cd = jnp.maximum(cd, 1.0e-3)
        # Calculate Driving Winds and Drag for Land (Scale by 0.7 for surface friction)
        u_surf = current_state.atmos.u * 0.7
        v_surf = current_state.atmos.v * 0.7
        wind_speed_mag = jnp.sqrt(u_surf**2 + v_surf**2) + 1.0
        
        # Calculate Seasonal Solar Forcing
        # Time [s] -> Day of Year
        seconds_in_year = 365.0 * 86400.0
        day_of_year = (current_state.time % seconds_in_year) / 86400.0
        
        lat_rad = jnp.deg2rad(jnp.linspace(-90, 90, ATMOS_GRID.nlat))
        sw_toa_profile = atmos_physics.compute_solar_insolation(lat_rad, day_of_year, solar_constant=params.solar_constant)
        
        # Broadcast 1D profile to 2D map
        sw_down = jnp.broadcast_to(sw_toa_profile[:, None], current_state.atmos.temp.shape)

        
        # LW Down still approx constant 300 or we can link to T_air
        # LW_down ~ epsilon_air * sigma * T_air^4
        # Approx T_air effective ~ T_air - 10K
        lw_down = 0.8 * 5.67e-8 * (current_state.atmos.temp - 10.0)**4
        
        # Peek call (precip=0)
        _, (land_sensible, land_latent, land_nee) = land_driver.step_land(
            current_state.land,
            t_air=current_state.atmos.temp,
            q_air=current_state.atmos.q,
            sw_down=sw_down,
            lw_down=lw_down,
            precip=jnp.zeros_like(current_state.atmos.temp),
            mask=land_mask,
            wind_speed=wind_speed_mag,
            drag_coeff=cd
        )
        
        # B. Ocean Fluxes
        # Use existing atmos_physics but masked for Ocean
        # Calculate Ocean Beta (Simple)
        beta_ocean = jnp.ones_like(current_state.atmos.temp)
        
        ocn_sens, ocn_lat = atmos_physics.compute_surface_fluxes(
            temp_air=current_state.atmos.temp,
            q_air=current_state.atmos.q,
            u_air=u_surf,
            v_air=v_surf,
            temp_surf=current_state.fluxes.sst, # Uses composite SST (includes Ice)
            beta=beta_ocean,
        )
        
        # C. Blend Fluxes
        # Atmos sees weighted average
        sensible_flux_coupled = land_mask * land_sensible + ocean_mask * ocn_sens
        latent_flux_coupled = land_mask * land_latent + ocean_mask * ocn_lat
        
        # Carbon Fluxes
        dic_surf = regridder.ocean_to_atmos(current_state.ocean.dic[0])
        pco2_sea = dic_surf * (280.0 / 2000.0)
        pco2_air = current_state.atmos.co2
        k_gas = 1.0e-8
        flux_c_sea = k_gas * (pco2_sea - pco2_air)
        flux_c_total = ocean_mask * flux_c_sea + land_mask * land_nee
        co2_flux_atm_ppm = flux_c_total * 240.0

        # 2. Step Atmosphere
        # ------------------
        # We are already inside the 'atmos_land_step' scan loop (outer loop).
        # So we just step ONCE here.
        
        new_atmos, (precip_atm, sfc_pressure) = atmos_driver.step_atmos(
            current_state.atmos,
            surface_temp=current_state.fluxes.sst,
            flux_sensible=sensible_flux_coupled,
            flux_latent=latent_flux_coupled,
            flux_co2=co2_flux_atm_ppm,
            sw_down=sw_down, # Pass calculated seasonal SW
            solar_constant=params.solar_constant,
            physics_params=physics_params,
            dt=DT_ATMOS,
            ny=ATMOS_GRID.nlat,
            nx=ATMOS_GRID.nlon,
        )
        # DEBUG
        # jax.debug.print("Step Precip Max: {}", jnp.max(precip_atm))
        
        # 3. Step Land (State Update)
        # ---------------------------
        # Now we have real Precip
        evap = latent_flux_coupled / 2.5e6
        fw_flux_atm = precip_atm - evap
        fw_flux_atm = jnp.clip(fw_flux_atm, -0.05, 0.05)
        
        precip_land_input = jnp.maximum(fw_flux_atm, 0.0) / 1000.0 # m/s
        
        new_land, _ = land_driver.step_land(
            current_state.land,
            t_air=current_state.atmos.temp, # Use consistent T_air (explicit)
            q_air=current_state.atmos.q,
            sw_down=sw_down,
            lw_down=lw_down,
            precip=precip_land_input,
            mask=land_mask,
            wind_speed=wind_speed_mag,
            drag_coeff=cd
        )
        
        # Update State Time
        new_time = current_state.time + DT_ATMOS
        
        # Calculate Net Heat for Ocean (Diagnostics/Coupling)
        # sw_down is already calculated (seasonal)
        # We need Surface Energy Balance for Ocean Driving
        # Net Heat = SW_net + LW_down - LW_up - Sensible - Latent
        
        # SW Net (Consistent with Atmosphere Albedo)
        # sw_down already includes (1-Albedo) * 0.52 from physics.py
        sw_net_ocean = sw_down
        
        # LW Up (Stefan-Boltzmann)
        sst_kelvin = current_state.fluxes.sst
        lw_up_ocean = 0.98 * 5.67e-8 * sst_kelvin**4
        
        # Net Heat
        net_heat_atm = sw_net_ocean + lw_down - lw_up_ocean - sensible_flux_coupled - latent_flux_coupled
        
        net_heat_atm = jnp.clip(net_heat_atm, -1500.0, 1500.0)
        
        # Update SST for next atmos step (Coupler from Ocean/Land to Atmos)
        # Ocean/Ice component is fixed during this loop, so ocean part of SST is constant
        # Land part of SST updates
        # Need to re-composite SST
        # Reconstruct SST (Ocean part + New Land part)
        # Note: current_state.fluxes.sst contains the composite. 
        # We need to extract the ocean part or remember it.
        # Ideally, we keep ocean_sst separate.
        # For simplicity, we assume ocean_temp in state is correct and constant during this loop.
        # We need to re-calculate sst_composite.
        
        # Get Ice surface temp (constant in this loop)
        # Ocean State is Celsius. SST State is Celsius.
        # DO NOT subtract 273.15 from Ocean Temp.
        sst_ocean_ice = (1.0 - current_state.ice.concentration) * (current_state.ocean.temp[0]) + \
                        current_state.ice.concentration * (current_state.ice.surface_temp + 273.15)
        
        # New Composite SST
        sst_composite = ocean_mask * sst_ocean_ice + land_mask * new_land.temp
        sst_atm = regridder.ocean_to_atmos(sst_composite)
        
        # Accumulate Fluxes for Ocean Driving
        # We need Net Heat, FW, Stress for Ocean
        # Net Heat to Ocean = Atmos Heat Flux (regridded)
        
        # Zonal Wind Stress
        lat = jnp.linspace(-90, 90, ATMOS_GRID.nlat)
        lat_rad = jnp.deg2rad(lat)
        
        # 1. Seasonality (ITCZ Shift +/- 10 degrees)
        # Shift profile center: lat_effective = lat - shift
        # Shift = 10 * sin(2*pi*day/365)
        # Summer (Day 180) -> Shift North (+10). Winter -> South.
        season_shift_deg = 10.0 * jnp.sin(2.0 * jnp.pi * (day_of_year - 80.0) / 365.0)
        season_shift_rad = jnp.deg2rad(season_shift_deg)
        
        lat_effective = lat_rad - season_shift_rad
        
        # Base Profile (Easterlies in Tropics)
        # Final Tuning: 0.08 Pa for realistic 15-25 Sv AMOC
        tau_base = -0.08 * jnp.sin(6.0 * lat_effective)
        
        # 2. Stochastic Noise (Pseudo-random based on time)
        # Use simple sine/cos chaotic mix to avoid passing PRNG key through 10 layers
        # A chaotic oscillation of amplitude 10%
        noise_factor = 0.1 * (jnp.sin(current_state.time / (86400.0 * 3.0)) + 
                              jnp.cos(current_state.time / (86400.0 * 7.0)))
        
        tau_profile = tau_base * (1.0 + noise_factor)
        
        tau_x_atm = jnp.broadcast_to(tau_profile[:, None], net_heat_atm.shape) # Simplified
        tau_y_atm = jnp.zeros_like(net_heat_atm)
        
        # Accumulate
        # Structure: (heat, fw, tau_x, tau_y, carbon, precip)
        acc_heat = acc_fluxes[0] + net_heat_atm
        acc_fw = acc_fluxes[1] + fw_flux_atm
        acc_tau_x = acc_fluxes[2] + tau_x_atm
        acc_tau_y = acc_fluxes[3] + tau_y_atm
        acc_carbon = acc_fluxes[4] + flux_c_sea
        acc_precip = acc_fluxes[5] + precip_atm
        
        new_acc = (acc_heat, acc_fw, acc_tau_x, acc_tau_y, acc_carbon, acc_precip)
        
        # Update Flux State for returning (instantaneous - maybe useful for debugging but we really want avg)
        # But wait, if we update precip here with instantaneous, we lose the avg.
        # We should update FluxState AFTER the loop with the average.
        # Inside the loop, we can keep instantaneous or just not update it yet?
        # The loop updates `current_state` which carries fluxes.
        # Let's keep instantaneous inside for now, but overwrite with Average after loop.
        
        new_fluxes = current_state.fluxes._replace(
            net_heat_flux=net_heat_atm,
            freshwater_flux=fw_flux_atm,
            sst=sst_atm,
            wind_stress_x=tau_x_atm,
            wind_stress_y=tau_y_atm,
            precip=precip_atm,
            carbon_flux_ocean=flux_c_sea,
            carbon_flux_land=land_nee
        )
        
        next_state = current_state._replace(
            atmos=new_atmos,
            land=new_land,
            fluxes=new_fluxes,
            time=new_time
        )
        
        return (next_state, new_acc), None

    # Initial Accumulator
    init_acc = (
        jnp.zeros_like(state.atmos.temp), # Heat
        jnp.zeros_like(state.atmos.temp), # FW
        jnp.zeros_like(state.atmos.temp), # Tau X
        jnp.zeros_like(state.atmos.temp), # Tau Y
        jnp.zeros_like(state.atmos.temp), # Carbon
        jnp.zeros_like(state.atmos.temp), # Precip
    )
    
    # Run Sub-stepping Loop
    (final_state_loop, accumulated_fluxes), _ = jax.lax.scan(
        atmos_land_step, (state, init_acc), None, length=n_substeps
    )
    
    # 2. Average Fluxes for Ocean
    avg_heat_atm = accumulated_fluxes[0] / n_substeps
    avg_fw_atm = accumulated_fluxes[1] / n_substeps
    avg_tau_x_atm = accumulated_fluxes[2] / n_substeps
    avg_tau_y_atm = accumulated_fluxes[3] / n_substeps
    avg_carbon_atm = accumulated_fluxes[4] / n_substeps
    avg_precip_atm = accumulated_fluxes[5] / n_substeps
    
    # Update FluxState with AVERAGES for output/consistency
    final_fluxes = final_state_loop.fluxes._replace(
        net_heat_flux=avg_heat_atm,
        freshwater_flux=avg_fw_atm,
        wind_stress_x=avg_tau_x_atm,
        wind_stress_y=avg_tau_y_atm,
        precip=avg_precip_atm,
        carbon_flux_ocean=avg_carbon_atm, # Should be avg/last? Avg makes sense.
        # carbon_flux_land was instantaneous in loop, but we should probably average it too if we tracked it.
        # For now, only precip is critical for this fix.
    )
    
    # Use final_state_loop but with averaged fluxes
    final_atmos_state = final_state_loop._replace(fluxes=final_fluxes)
    
    # Regrid to Ocean
    heat_flux_ocn = regridder.atmos_to_ocean(avg_heat_atm)
    fw_flux_ocn = regridder.atmos_to_ocean(avg_fw_atm)
    tau_x_ocn = regridder.atmos_to_ocean(avg_tau_x_atm)
    tau_y_ocn = regridder.atmos_to_ocean(avg_tau_y_atm)
    
    # 3. Sea Ice Step
    # Use final atmospheric state for T_air (snapshot coupling for thermodynamics is ok)
    t_air_ocn = regridder.atmos_to_ocean(final_atmos_state.atmos.temp - 273.15)
    sw_down = jnp.zeros_like(t_air_ocn)
    lw_down = jnp.ones_like(t_air_ocn) * 300.0
    sst_ocean_grid = final_atmos_state.ocean.temp[0] - 273.15
    
    ocean_mask = params.mask if (hasattr(params, "mask") and params.mask is not None) else None
    
    new_ice, (ice_heat_flux, ice_fw_flux) = ice_driver.step_ice(
        final_atmos_state.ice,
        t_air=t_air_ocn,
        sw_down=sw_down,
        lw_down=lw_down,
        ocean_temp=sst_ocean_grid,
        ny=OCEAN_GRID.nlat,
        nx=OCEAN_GRID.nlon,
        mask=ocean_mask,
    )

    # 4. Ocean Step
    A = new_ice.concentration
    combined_heat_flux = (1.0 - A) * heat_flux_ocn + A * ice_heat_flux
    combined_fw_flux = (1.0 - A) * fw_flux_ocn + A * ice_fw_flux

    fluxes_ocean = (combined_heat_flux, combined_fw_flux, -avg_carbon_atm)
    wind_ocean = (tau_x_ocn, tau_y_ocn)

    # Calculate grid spacing [m]
    # T31: nlat=48, nlon=96
    # dx = 2*pi*R * cos(lat) / nlon
    # dy = pi*R / nlat (approx)
    
    # Veros driver currently expects scalar dx, dy.
    # We will use the spacing at 45 degrees as a representative constant for this version,
    # or ideally update veros_driver to accept arrays. 
    # For now, we fix the "100km" bug by using a realistic average.
    # At 45 deg: cos(45) ~ 0.707.
    # dx ~ 2*pi*6.371e6 * 0.707 / 96 ~ 295 km.
    # dy ~ pi*6.371e6 / 48 ~ 417 km.
    
    # Better: Use Equatorial dx (~417km) and dy (~417km) or a mean.
    # Let's use the explicit formulas.
    
    from chronos_esm.config import EARTH_RADIUS
    
    # dy is constant in latitude
    dy_ocn = (jnp.pi * EARTH_RADIUS) / OCEAN_GRID.nlat
    
    # Calculate latitude-dependent dx array
    lat_ocn = jnp.linspace(-90, 90, OCEAN_GRID.nlat)
    cos_lat_ocn = jnp.cos(jnp.deg2rad(lat_ocn))
    # Avoid zero dx at poles for numerical stability in diffusion
    cos_lat_ocn = jnp.maximum(cos_lat_ocn, 0.05) 
    dx_ocn_array = (2 * jnp.pi * EARTH_RADIUS * cos_lat_ocn[:, None]) / OCEAN_GRID.nlon
    
    # Vertical Grid (dz)
    # Must match data.load_initial_conditions: linspace(0, 5000, 15) -> 14 intervals of ~357m ?
    # Wait, load_initial_conditions does linspace(0, 5000, 15) for interpolation target.
    # Usually this defines the cell centers or interfaces.
    # If 15 levels, total depth 5000m.
    # dz = 5000 / 15 = 333.33 m.
    # Let's use this to match total depth ~5000m.
    dz_ocn = jnp.ones(state.ocean.u.shape[0]) * (5000.0 / state.ocean.u.shape[0])
    
    # Step Ocean (DT_OCEAN)
    new_ocean = ocean_driver.step_ocean(
        final_atmos_state.ocean,
        surface_fluxes=fluxes_ocean,
        wind_stress=wind_ocean,
        dx=dx_ocn_array,
        dy=dy_ocn,
        dz=dz_ocn,
        nz=state.ocean.u.shape[0],
        ny=OCEAN_GRID.nlat,
        nx=OCEAN_GRID.nlon,
        mask=ocean_mask,
        dt=DT_OCEAN,
        r_drag=r_drag, # Use passed parameter
        kappa_gm=kappa_gm, # Use passed parameter
        kappa_h=kappa_h,
        kappa_bi=kappa_bi,
        Ah=Ah,

        Ab=Ab,
        shapiro_strength=shapiro_strength,
        smag_constant=smag_constant
    )
    
    # Apply Bathymetry Mask
    # Apply Bathymetry Mask
    if ocean_mask is not None:
        mask_3d = jnp.broadcast_to(ocean_mask, new_ocean.u.shape)
        u_masked = jnp.where(mask_3d, new_ocean.u, 0.0)
        v_masked = jnp.where(mask_3d, new_ocean.v, 0.0)
        new_ocean = new_ocean._replace(u=u_masked, v=v_masked)

    # Removed hard crash logic to allow differentiable tuning





    # Update Coupling State (New Ocean, New Ice, Final Atmos/Land)
    
    # Re-calculate composite SST for FluxState/Next Step consistency
    sst_ocean_ice = (1.0 - A) * (new_ocean.temp[0]) + A * (new_ice.surface_temp + 273.15)
    
    land_mask = 1.0 - ocean_mask if ocean_mask is not None else 0.0
    if ocean_mask is not None:
         sst_composite = ocean_mask * sst_ocean_ice + land_mask * final_atmos_state.land.temp
    else:
         sst_composite = sst_ocean_ice
         
    sst_atm = regridder.ocean_to_atmos(sst_composite)
    
    # Update fluxes structure with newest state
    final_fluxes = final_atmos_state.fluxes._replace(sst=sst_atm)

    return coupled_state.CoupledState(
        ocean=new_ocean,
        atmos=final_atmos_state.atmos,
        ice=new_ice,
        land=final_atmos_state.land,
        fluxes=final_fluxes,
        time=final_atmos_state.time,
    )


def run_simulation(
    steps: int, params: ModelParams = ModelParams()
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
