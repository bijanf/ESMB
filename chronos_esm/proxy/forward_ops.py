"""
Forward operators for proxy modeling.

Maps model state variables (e.g., SST, Salinity) to proxy observations (e.g., d18O).
"""

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


# Global surface salinity--d18O_sw relationship. The slope ~0.5 permil/psu is the
# global-ocean surface mean of LeGrande & Schmidt (2006); we reference it to the
# global-mean salinity S_ref = 35 psu so that d18O_sw = 0 (VSMOW) at S_ref.
# Regional slopes differ markedly (e.g. high-latitude meltwater lines are steeper);
# pass `slope`/`s_ref` to override for a basin-specific calibration.
D18OSW_SLOPE = 0.5  # permil per psu (LeGrande & Schmidt 2006, global surface mean)
D18OSW_S_REF = 35.0  # psu (reference salinity, d18O_sw = 0 here)


def salinity_to_d18o_sw(
    salinity: jnp.ndarray,
    slope: float = D18OSW_SLOPE,
    s_ref: float = D18OSW_S_REF,
) -> jnp.ndarray:
    """
    Linear seawater d18O--salinity relationship (LeGrande & Schmidt 2006).

        d18O_sw = slope * (S - s_ref)

    The default global slope (0.5 permil/psu, referenced to S_ref = 35 psu) is a
    first-order surface approximation; supply `slope`/`s_ref` for a regional
    calibration. Fully differentiable in the input salinity and the coefficients.

    Args:
        salinity: Practical salinity [psu]
        slope: d(d18O_sw)/dS [permil/psu] (default global-mean 0.5)
        s_ref: reference salinity at which d18O_sw = 0 [psu] (default 35)

    Returns:
        d18o_sw: Seawater d18O [permil, VSMOW]
    """
    return slope * (salinity - s_ref)
