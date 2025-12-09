"""
Input/Output module for Chronos-ESM.

Handles saving model state to NetCDF.
"""

from pathlib import Path
from typing import Union

import jax.numpy as jnp
import numpy as np
import xarray as xr

from chronos_esm.config import ATMOS_GRID, OCEAN_GRID
from chronos_esm.coupler import state as coupled_state


def save_state_to_netcdf(state: coupled_state.CoupledState, filepath: Union[str, Path]):
    """
    Save the coupled state to a NetCDF file.

    Args:
        state: The coupled model state.
        filepath: Path to save the file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert JAX arrays to numpy
    # Ocean
    temp_ocn = np.array(state.ocean.temp)
    salt_ocn = np.array(state.ocean.salt)
    u_ocn = np.array(state.ocean.u)
    v_ocn = np.array(state.ocean.v)
    w_ocn = np.array(state.ocean.w)

    # Ice
    h_ice = np.array(state.ice.thickness)
    a_ice = np.array(state.ice.concentration)

    # Atmos
    temp_atm = np.array(state.atmos.temp)
    q_atm = np.array(state.atmos.q)
    u_atm = np.array(state.atmos.u)
    v_atm = np.array(state.atmos.v)

    # Coordinates
    nz, ny_ocn, nx_ocn = temp_ocn.shape
    ny_atm, nx_atm = temp_atm.shape

    # Create Dataset
    ds = xr.Dataset(
        data_vars={
            # Ocean
            "ocean_temp": (
                ("z", "y_ocn", "x_ocn"),
                temp_ocn,
                {"units": "K", "long_name": "Ocean Potential Temperature"},
            ),
            "ocean_salt": (
                ("z", "y_ocn", "x_ocn"),
                salt_ocn,
                {"units": "psu", "long_name": "Ocean Salinity"},
            ),
            "ocean_u": (
                ("z", "y_ocn", "x_ocn"),
                u_ocn,
                {"units": "m/s", "long_name": "Ocean Zonal Velocity"},
            ),
            "ocean_v": (
                ("z", "y_ocn", "x_ocn"),
                v_ocn,
                {"units": "m/s", "long_name": "Ocean Meridional Velocity"},
            ),
            # Ice
            "ice_thickness": (
                ("y_ocn", "x_ocn"),
                h_ice,
                {"units": "m", "long_name": "Sea Ice Thickness"},
            ),
            "ice_concentration": (
                ("y_ocn", "x_ocn"),
                a_ice,
                {"units": "1", "long_name": "Sea Ice Concentration"},
            ),
            # Atmos
            "atmos_temp": (
                ("y_atm", "x_atm"),
                temp_atm,
                {"units": "K", "long_name": "Atmospheric Temperature"},
            ),
            "atmos_q": (
                ("y_atm", "x_atm"),
                q_atm,
                {"units": "kg/kg", "long_name": "Specific Humidity"},
            ),
            "atmos_u": (
                ("y_atm", "x_atm"),
                u_atm,
                {"units": "m/s", "long_name": "Atmospheric Zonal Wind"},
            ),
            "atmos_v": (
                ("y_atm", "x_atm"),
                v_atm,
                {"units": "m/s", "long_name": "Atmospheric Meridional Wind"},
            ),
            # Time
            "time": ((), float(state.time), {"units": "seconds since start"}),
        },
        coords={
            "z": np.arange(nz),
            "y_ocn": np.arange(ny_ocn),
            "x_ocn": np.arange(nx_ocn),
            "y_atm": np.arange(ny_atm),
            "x_atm": np.arange(nx_atm),
        },
        attrs={
            "description": "Chronos-ESM Coupled State",
            "model_time": float(state.time),
        },
    )

    ds.to_netcdf(filepath)
    print(f"Saved state to {filepath}")
