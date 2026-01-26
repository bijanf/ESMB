"""
Global configuration for Chronos-ESM.
Dynmaic Resolution Support (T31/T63).
"""

import os
from typing import NamedTuple

import jax.numpy as jnp  # noqa: F401

# ============================================================================
# Resolution Configuration
# ============================================================================

# Default to T31 if not specified
RESOLUTION = os.environ.get("CHRONOS_RESOLUTION", "T31").upper()

if RESOLUTION == "T63":
    NLAT = 96
    NLON = 192
    DEFAULT_DT_ATMOS = 30.0  # Safe for T63 with proper diffusion
elif RESOLUTION == "T31":
    NLAT = 48
    NLON = 96
    DEFAULT_DT_ATMOS = 30.0  # Reverted to 30s for stability (Safe Baseline)
else:
    raise ValueError(f"Unknown resolution: {RESOLUTION}. Supported: T31, T63")

# Atmospheric grid
NLAT_ATMOS = NLAT
NLON_ATMOS = NLON

# Ocean grid
NLAT_OCEAN = NLAT
NLON_OCEAN = NLON
NZ_OCEAN = 15  # Default vertical levels

# ============================================================================
# Physical Constants
# ============================================================================

EARTH_RADIUS = 6.371e6  # Earth radius [m]
GRAVITY = 9.81  # Gravitational acceleration [m/s^2]
OMEGA = 7.2921e-5  # Earth's angular velocity [rad/s]
RADIUS_EARTH = EARTH_RADIUS

RHO_WATER = 1025.0  # Seawater density [kg/m^3]
RHO_AIR = 1.225  # Air density at sea level [kg/m^3]
CP_AIR = 1004.0  # Specific heat of air [J/kg/K]
R_AIR = 287.0 # Gas constant for dry air [J/kg/K]
CP_WATER = 3985.0  # Specific heat of seawater [J/kg/K]
LATENT_HEAT_VAPORIZATION = 2.5e6  # Latent heat of vaporization [J/kg]
STEFAN_BOLTZMANN = 5.67e-8  # Stefan-Boltzmann constant [W/m^2/K^4]
P0 = 100000.0  # Reference pressure [Pa]
KAPPA = R_AIR / CP_AIR # Ratio of gas constant to specific heat at constant pressure

# Tuning Parameters
DRAG_COEFF_OCEAN = 1.2e-3
DRAG_COEFF_LAND = 2.0e-3  # Higher roughness over land
ALBEDO_OCEAN = 0.07
ALBEDO_LAND = 0.3
EMISSIVITY_LAND = 0.96

# ============================================================================
# Numerical Parameters
# ============================================================================

# Time Stepping
DT_OCEAN = 900.0  # Ocean timestep [s] (15 mins)
# Allow override of DT via env var for tuning
DT_ATMOS = float(os.environ.get("CHRONOS_DT_ATMOS", DEFAULT_DT_ATMOS))
COUPLING_INTERVAL = 3600.0  # Exchange interval [s] (1 hour)

# Solver tolerances
CG_TOL = 1e-6  # Conjugate Gradient tolerance
CG_MAX_ITER = 1000  # Maximum CG iterations

# Smoothing parameters for differentiable physics
EPSILON_SMOOTH = 1e-3  # Softplus sharpness for precipitation


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
