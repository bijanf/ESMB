"""
Dynamic Vegetation Module.

Implements a simple prognostic vegetation model based on Leaf Area Index (LAI).
Vegetation grows based on temperature and soil moisture availability.
"""

from typing import Tuple

import jax
import jax.numpy as jnp

from chronos_esm.config import DT_ATMOS

# Vegetation Constants
LAI_MAX = 6.0  # Maximum Leaf Area Index
LAI_MIN = 0.1  # Minimum LAI (bare soil/desert)
GROWTH_TIMESCALE = 86400.0 * 10.0  # 10 days e-folding growth
DECAY_TIMESCALE = 86400.0 * 30.0  # 30 days decay (senescence)

# Growth constraints
TEMP_MIN_GROWTH = 278.15  # 5 deg C
TEMP_OPT_GROWTH = 298.15  # 25 deg C
TEMP_MAX_GROWTH = 313.15  # 40 deg C

# Albedo
ALBEDO_SOIL = 0.35
ALBEDO_VEG = 0.12  # Darker than soil


def compute_growth_factor(
    temp: jnp.ndarray, moisture_stress: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute growth potential (0 to 1) based on environmental conditions.

    Args:
        temp: Surface temperature [K]
        moisture_stress: Soil moisture availability factor (0-1)

    Returns:
        growth_rate: Normalized growth potential (0-1)
    """
    # Temperature function (parabolic)
    # T_factor = (T - T_min) * (T_max - T) / ((T_opt - T_min) * (T_max - T_opt))
    # Simplified: Gaussian-like bell curve around T_opt
    t_sigma = 15.0
    t_factor = jnp.exp(-((temp - TEMP_OPT_GROWTH) ** 2) / (2 * t_sigma**2))

    # Hard cutoff for freezing
    t_factor = jnp.where(temp < TEMP_MIN_GROWTH, 0.0, t_factor)

    # Combined factor
    return t_factor * moisture_stress


def step_vegetation(
    lai: jnp.ndarray, temp: jnp.ndarray, soil_moisture: jnp.ndarray, bucket_depth: float
) -> jnp.ndarray:
    """
    Time step vegetation state (LAI).

    dLAI/dt = Growth - Decay
    Growth = r * LAI * (1 - LAI/LAI_max) * GrowthFactor
    Decay = LAI / tau_decay

    Args:
        lai: Current Leaf Area Index
        temp: Surface temperature [K]
        soil_moisture: Soil moisture depth [m]
        bucket_depth: Max soil moisture depth [m]

    Returns:
        lai_new: Updated LAI
    """
    # Moisture stress factor (beta)
    beta = soil_moisture / bucket_depth
    beta = jnp.clip(beta, 0.0, 1.0)

    # Growth potential
    growth_potential = compute_growth_factor(temp, beta)

    # Logistic Growth
    # We add a small seed growth so bare soil can recover
    growth_rate = (1.0 / GROWTH_TIMESCALE) * growth_potential
    growth_term = growth_rate * lai * (1.0 - lai / LAI_MAX)

    # Seed term (spontaneous growth if conditions are good)
    seed_term = 1e-8 * growth_potential

    # Decay (senescence + stress)
    # Decay increases if conditions are bad (e.g. drought)
    stress_decay = (1.0 - beta) * (1.0 / (DECAY_TIMESCALE * 0.5))
    background_decay = 1.0 / DECAY_TIMESCALE

    total_decay = (background_decay + stress_decay) * lai

    # Update
    dlai_dt = growth_term + seed_term - total_decay

    lai_new = lai + DT_ATMOS * dlai_dt

    # Bounds
    lai_new = jnp.clip(lai_new, LAI_MIN, LAI_MAX)

    return lai_new


def compute_land_properties(lai: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute physical properties based on LAI.

    Args:
        lai: Leaf Area Index

    Returns:
        albedo: Surface albedo
        veg_fraction: Fraction of ground covered by vegetation
    """
    # Vegetation fraction (Beer-Lambert law approximation)
    # f_veg = 1 - exp(-k * LAI), k ~ 0.5
    veg_fraction = 1.0 - jnp.exp(-0.5 * lai)

    # Dynamic Albedo
    # Mixed albedo = f_veg * alpha_veg + (1 - f_veg) * alpha_soil
    albedo = veg_fraction * ALBEDO_VEG + (1.0 - veg_fraction) * ALBEDO_SOIL

    return albedo, veg_fraction
