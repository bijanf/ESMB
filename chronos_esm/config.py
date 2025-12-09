"""
Global configuration for Chronos-ESM.
T63 Resolution Earth System Model.
"""

from typing import NamedTuple

import jax.numpy as jnp  # noqa: F401

# ============================================================================
# Grid Definitions (T63 Resolution)
# ============================================================================

# Atmospheric grid (Gaussian)
NLAT_ATMOS = 96  # Number of latitudes
NLON_ATMOS = 192  # Number of longitudes

# Ocean grid (T63 equivalent, ~1.875° resolution)
NLAT_OCEAN = 96
NLON_OCEAN = 192
NZ_OCEAN = 40  # Vertical levels

# ============================================================================
# Physical Constants
# ============================================================================

EARTH_RADIUS = 6.371e6  # Earth radius [m]
GRAVITY = 9.81  # Gravitational acceleration [m/s^2]
OMEGA = 7.2921e-5  # Earth's angular velocity [rad/s]
RHO_WATER = 1025.0  # Seawater density [kg/m^3]
RHO_AIR = 1.225  # Air density at sea level [kg/m^3]
CP_AIR = 1004.0  # Specific heat of air [J/kg/K]
CP_WATER = 3985.0  # Specific heat of seawater [J/kg/K]
LATENT_HEAT_VAPORIZATION = 2.5e6  # Latent heat of vaporization [J/kg]
STEFAN_BOLTZMANN = 5.67e-8  # Stefan-Boltzmann constant [W/m^2/K^4]

# Tuning Parameters
DRAG_COEFF_OCEAN = 1.2e-3
DRAG_COEFF_LAND = 2.0e-3  # Higher roughness over land
ALBEDO_OCEAN = 0.07
ALBEDO_LAND = 0.3
EMISSIVITY_LAND = 0.96

# ============================================================================
# Numerical Parameters
# ============================================================================

DT_OCEAN = 3600.0  # Ocean timestep [s] (1 hour)
DT_OCEAN = 3600.0  # Ocean timestep [s] (1 hour)
DT_ATMOS = 120.0  # Atmosphere timestep [s] (Optimized for stability/speed)
COUPLING_INTERVAL = 3600.0  # Exchange interval [s] (1 hour)

# Solver tolerances
CG_TOL = 1e-6  # Conjugate Gradient tolerance
CG_MAX_ITER = 1000  # Maximum CG iterations

# Smoothing parameters for differentiable physics
EPSILON_SMOOTH = 1e-5  # Softplus sharpness for precipitation


class GridConfig(NamedTuple):
    """Configuration for a spatial grid."""

    nlat: int
    nlon: int
    nz: int = 1

    @property
    def shape_2d(self):
        return (self.nlat, self.nlon)

    @property
    def shape_3d(self):
        return (self.nlat, self.nlon, self.nz)

    @property
    def size(self):
        return self.nlat * self.nlon * self.nz


# Pre-defined grid configurations
ATMOS_GRID = GridConfig(nlat=NLAT_ATMOS, nlon=NLON_ATMOS, nz=1)
OCEAN_GRID = GridConfig(nlat=NLAT_OCEAN, nlon=NLON_OCEAN, nz=NZ_OCEAN)
