"""
Coupled state definitions.
"""

from typing import NamedTuple, Tuple
import jax.numpy as jnp

from chronos_esm.ocean.veros_driver import OceanState
from chronos_esm.atmos.dynamics import AtmosState

class FluxState(NamedTuple):
    """
    Exchange fluxes between components.
    """
    # Atmos -> Ocean
    net_heat_flux: jnp.ndarray      # [W/m^2] (positive down)
    freshwater_flux: jnp.ndarray    # [kg/m^2/s] (P - E)
    wind_stress_x: jnp.ndarray      # [N/m^2]
    wind_stress_y: jnp.ndarray      # [N/m^2]
    precip: jnp.ndarray             # [kg/m^2/s] Precipitation rate
    
    # Ocean -> Atmos
    sst: jnp.ndarray                # Sea Surface Temperature [K]
    
    # Carbon Fluxes (Positive Upward = Release to Atmos) [kg C/m2/s]
    carbon_flux_ocean: jnp.ndarray
    carbon_flux_land: jnp.ndarray


from chronos_esm.ice.driver import IceState, init_ice_state
from chronos_esm.land.driver import LandState, init_land_state

class CoupledState(NamedTuple):
    """
    Full state of the Earth System.
    """
    ocean: OceanState
    atmos: AtmosState
    ice: IceState
    land: LandState
    fluxes: FluxState
    time: float                     # Simulation time [s]


def init_coupled_state(ocean_state: OceanState, atmos_state: AtmosState, land_state: LandState = None) -> CoupledState:
    """Initialize coupled state."""
    ny, nx = atmos_state.temp.shape
    
    # Initial fluxes (zero)
    fluxes = FluxState(
        net_heat_flux=jnp.zeros((ny, nx)),
        freshwater_flux=jnp.zeros((ny, nx)),
        wind_stress_x=jnp.zeros((ny, nx)),
        wind_stress_y=jnp.zeros((ny, nx)),
        precip=jnp.zeros((ny, nx)),
        sst=jnp.ones((ny, nx)) * 290.0, # Initial SST guess
        carbon_flux_ocean=jnp.zeros((ny, nx)),
        carbon_flux_land=jnp.zeros((ny, nx))
    )
    
    # Initialize Ice
    ice_state = init_ice_state(ny, nx)
    
    # Initialize Land if not provided
    if land_state is None:
        land_state = init_land_state(ny, nx)
    
    return CoupledState(
        ocean=ocean_state,
        atmos=atmos_state,
        ice=ice_state,
        land=land_state,
        fluxes=fluxes,
        time=0.0
    )
