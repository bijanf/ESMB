"""
Forcing Manager for Chronos-ESM.

Handles historical forcing data (CO2, Solar, Aerosols) for reconstruction experiments.
Uses simple interpolation of historical records.
"""

from typing import NamedTuple, Optional

import jax.numpy as jnp
import numpy as np

from chronos_esm.main import ModelParams


class ForcingData(NamedTuple):
    year: jnp.ndarray
    co2: jnp.ndarray  # ppm
    solar: jnp.ndarray  # W/m^2
    volcanic: jnp.ndarray  # Optical Depth (AOD) - Optional


class ForcingManager:
    """Manages time-dependent forcing parameters."""

    def __init__(self):
        # Historical CO2 (approximate annual means)
        # Source: IPCC / NOAA
        self.years = jnp.array([
            1850.0, 1900.0, 1950.0, 1960.0, 1970.0, 1980.0, 1990.0, 2000.0, 2010.0, 2020.0, 2025.0
        ])
        
        self.co2_vals = jnp.array([
            285.2, 295.7, 311.3, 316.9, 325.6, 338.7, 354.3, 369.5, 389.9, 414.2, 423.0
        ])

        # Solar Irradiance (TSI) - simplified cycle or constant
        # For reconstruction, we might want the 11-year cycle superimposed
        # Base: 1361.0
        self.solar_base = 1361.0
    
    def get_forcing(self, year: float) -> ModelParams:
        """
        Get model parameters for a specific decimal year.
        
        Args:
            year: Decimal year (e.g. 1950.5)
            
        Returns:
            ModelParams with interpolated CO2 and Solar
        """
        # Interpolate CO2
        co2 = jnp.interp(year, self.years, self.co2_vals)
        
        # Solar Cycle (Simple Sinusoid approx)
        # Amplitude ~1 W/m2 (0.1%)
        # Period 11 years
        phase = 2 * jnp.pi * (year - 1850) / 11.0
        solar = self.solar_base + 0.5 * jnp.sin(phase)
        
        # Volcanic? (Maybe later)
        
        return ModelParams(
            co2_ppm=co2,
            solar_constant=solar
        )

# Global Instance
forcing_manager = ForcingManager()
