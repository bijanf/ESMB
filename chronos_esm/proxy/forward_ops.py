"""
Forward operators for proxy modeling.

Maps model state variables (e.g., SST, Salinity) to proxy observations (e.g., d18O).
"""

import jax
import jax.numpy as jnp

# Constants for Bemis et al. (1998)
# T = 16.5 - 4.80 * (d18Oc - d18Osw)
# Inverted: d18Oc = d18Osw + (16.5 - T) / 4.80
BEMIS_A = 16.5
BEMIS_B = 4.80

def bemis_d18o(sst_c: jnp.ndarray, d18o_sw: jnp.ndarray = 0.0) -> jnp.ndarray:
    """
    Forward operator for planktonic foraminifera d18O using Bemis et al. (1998).
    
    Args:
        sst_c: Sea Surface Temperature [C]
        d18o_sw: Seawater d18O [permil] (default 0.0 or passed from salinity)
        
    Returns:
        d18o_c: Calcite d18O [permil]
    """
    return d18o_sw + (BEMIS_A - sst_c) / BEMIS_B


def salinity_to_d18o_sw(salinity: jnp.ndarray) -> jnp.ndarray:
    """
    Simple linear relationship between Salinity and d18O_sw.
    
    d18O_sw = alpha * S + beta
    Global relationship approx: d18O_sw = 0.5 * S - 17.5 (rough approx)
    Or regional.
    
    For now, return 0.0 or a simple placeholder.
    """
    # Placeholder: assume constant 0.0 SMOW for now unless requested
    return jnp.zeros_like(salinity)
