"""
Driver for the Sea Ice component.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple
from functools import partial

from chronos_esm.config import OCEAN_GRID, DT_OCEAN
from chronos_esm.ice import thermodynamics

class IceState(NamedTuple):
    """State of the sea ice."""
    thickness: jnp.ndarray   # Ice thickness [m] (h)
    concentration: jnp.ndarray # Ice concentration [0-1] (A)
    surface_temp: jnp.ndarray # Surface temperature [C]


def init_ice_state(ny, nx) -> IceState:
    """Initialize no ice."""
    return IceState(
        thickness=jnp.zeros((ny, nx)),
        concentration=jnp.zeros((ny, nx)),
        surface_temp=jnp.ones((ny, nx)) * -1.8
    )


@partial(jax.jit, static_argnames=['ny', 'nx'])
def step_ice(
    state: IceState,
    t_air: jnp.ndarray,
    sw_down: jnp.ndarray,
    lw_down: jnp.ndarray,
    ocean_temp: jnp.ndarray, # SST
    ny: int = OCEAN_GRID.nlat,
    nx: int = OCEAN_GRID.nlon,
    mask: jnp.ndarray = None # 1=Ocean, 0=Land
) -> Tuple[IceState, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Time step sea ice.
    
    Args:
        state: Current ice state
        t_air: Air temperature [C]
        sw_down: Shortwave flux
        lw_down: Longwave flux
        ocean_temp: Sea Surface Temperature [C]
        mask: Land mask (1=Ocean, 0=Land)
        
    Returns:
        new_state: Updated ice state
        fluxes: (heat_flux_to_ocean, freshwater_flux)
    """
    # Constants
    rho_ice = thermodynamics.RHO_ICE
    L_fusion = thermodynamics.L_FUSION
    k_ice = thermodynamics.K_ICE
    t_freeze = thermodynamics.T_FREEZE
    
    # 1. Thermodynamics
    # Solve surface temperature
    # Only solve where ice exists (h > 0.01)
    has_ice = state.thickness > 0.01
    
    # For open water, Ts is SST (or T_freeze if freezing)
    # We solve for Ts everywhere but mask later
    ts, net_surf_flux = thermodynamics.solve_surface_temp(
        t_air, sw_down, lw_down, state.thickness, t_freeze
    )
    
    # 2. Growth/Melt
    # Bottom accretion: k*(Tf - Ts)/h
    # Only if ice exists
    h_safe = jnp.maximum(state.thickness, 0.1)
    conduction_flux = k_ice * (t_freeze - ts) / h_safe
    
    # If Ts < 0, net_surf_flux is 0 (balanced).
    # If Ts = 0, net_surf_flux > 0 (melting energy).
    # Top melt: F_surf / (rho * L)
    melt_top = jnp.maximum(net_surf_flux, 0.0) / (rho_ice * L_fusion)
    
    # Bottom growth: F_cond / (rho * L)
    # Note: Ocean heat flux subtracts from this. Assuming 0 for now.
    growth_bottom = conduction_flux / (rho_ice * L_fusion)
    
    # Total thickness change
    dh_dt = growth_bottom - melt_top
    
    # Apply change
    h_new = state.thickness + DT_OCEAN * dh_dt
    
    # 3. New Ice Formation (Frazil)
    # If SST < T_freeze, form ice
    # Energy to freeze: Cp_water * (T_freeze - SST) * rho_water * depth?
    # Simplified: relax SST to T_freeze and convert deficit to ice
    # Max supercooling?
    supercooling = jnp.maximum(t_freeze - ocean_temp, 0.0)
    # Convert supercooling to ice thickness
    # Energy = rho_w * cw * supercooling * mixed_layer_depth
    # Ice = Energy / (rho_i * L)
    mld = 50.0 # meters
    rho_w = 1025.0
    cw = 3985.0
    
    new_ice = (rho_w * cw * supercooling * mld) / (rho_ice * L_fusion)
    
    # Add new ice
    h_new = h_new + new_ice
    
    # Apply Mask (No ice on land)
    if mask is not None:
        h_new = h_new * mask
        new_ice = new_ice * mask # Ensure flux calculation is also masked
    
    # 4. Concentration (Hibler approx)
    # A = h / h_0? Or advection?
    # Simple: if h > 0, A = 1. Or A relaxes to 1.
    # Let's use A = tanh(h / 0.5)
    a_new = jnp.tanh(h_new / 0.5)
    
    # Clip
    h_new = jnp.maximum(h_new, 0.0)
    a_new = jnp.maximum(a_new, 0.0)
    
    # Update Ts (only valid where ice exists)
    ts_new = jnp.where(h_new > 0.01, ts, ocean_temp)
    
    new_state = IceState(
        thickness=h_new,
        concentration=a_new,
        surface_temp=ts_new
    )
    
    # Fluxes to Ocean
    # Heat flux:
    # 1. Conduction through ice (warming ocean? No, cooling ocean/freezing)
    #    Actually, conduction removes heat from ocean interface -> ice growth.
    #    So heat flux to ocean is NEGATIVE (loss).
    # 2. Frazil formation releases latent heat to ocean (warming it back to T_freeze).
    #    So heat flux to ocean is POSITIVE.
    
    # Let's define heat_flux_to_ocean as energy passed from atmos/ice to ocean.
    # If ice grows at bottom, it extracts heat: -conduction_flux
    # If frazil forms, it adds heat (latent): +new_ice * rho * L / dt
    
    # Also, if ice exists, ocean is insulated from atmos fluxes (handled in coupler).
    # Here we return the ice-ocean exchange.
    
    flux_conduct = -conduction_flux # W/m^2 (extracting heat)
    flux_frazil = (new_ice * rho_ice * L_fusion) / DT_OCEAN
    
    net_heat_flux = flux_conduct + flux_frazil
    
    # Freshwater flux
    # Growth = salt rejection (brine) -> Salinity increases -> FW flux negative
    # Melt = fresh water input -> FW flux positive
    # Flux = - (rho_i / rho_w) * dh/dt
    # Wait, FW flux usually defined as mass of fresh water.
    # Melt (dh/dt < 0) -> Positive FW flux.
    # Growth (dh/dt > 0) -> Negative FW flux.
    
    fw_flux = -(rho_ice) * (h_new - state.thickness) / DT_OCEAN
    
    return new_state, (net_heat_flux, fw_flux)
