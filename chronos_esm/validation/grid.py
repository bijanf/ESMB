"""
Model grid helpers and regridding of observations onto the model grid.

The model uses a regular lat-lon grid:
    lat = linspace(-90, 90, nlat)              (includes the poles)
    lon = linspace(-180, 180, nlon, endpoint=False)
matching chronos_esm.data.load_initial_conditions. Area weights are cos(lat).

Regridding uses scipy's RegularGridInterpolator (bilinear) with periodic
padding in longitude, mirroring chronos_esm.data — no xesmf/cartopy needed.
NaNs in the source (e.g. land in WOA) propagate to the model grid, where the
metrics treat them as missing.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from chronos_esm.data import pad_periodic_longitude


def model_lat(nlat):
    """Model latitudes [deg], -90..90 inclusive."""
    return np.linspace(-90.0, 90.0, nlat)


def model_lon(nlon):
    """Model longitudes [deg], -180..180 (endpoint excluded)."""
    return np.linspace(-180.0, 180.0, nlon, endpoint=False)


def area_weights(lat):
    """cos(lat) area weights as a (nlat, 1) column for broadcasting."""
    lat = np.asarray(lat, dtype=float)
    w = np.cos(np.deg2rad(lat))
    w = np.clip(w, 0.0, None)
    return w[:, None]


def regrid_to_model(field, src_lat, src_lon, dst_lat, dst_lon):
    """Bilinearly regrid a 2-D field (src_lat, src_lon) onto (dst_lat, dst_lon).

    Source longitudes are normalized to [-180, 180) and sorted; source
    latitudes are flipped to ascending if needed; longitude is periodically
    padded so the dateline interpolates correctly. Points outside the source
    latitude range (e.g. true poles vs a source that stops at +-87.5) get NaN.
    """
    field = np.asarray(field, dtype=float)
    src_lat = np.asarray(src_lat, dtype=float)
    src_lon = np.asarray(src_lon, dtype=float)

    # Normalize source longitude to [-180, 180) and sort columns to match.
    src_lon = ((src_lon + 180.0) % 360.0) - 180.0
    order = np.argsort(src_lon)
    src_lon = src_lon[order]
    field = field[..., order]

    # Ascending latitude.
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        field = field[..., ::-1, :]

    field_p, lon_p = pad_periodic_longitude(field, src_lon)

    interp = RegularGridInterpolator(
        (src_lat, lon_p), field_p,
        method="linear", bounds_error=False, fill_value=np.nan,
    )

    dst_lat = np.asarray(dst_lat, dtype=float)
    dst_lon = np.asarray(dst_lon, dtype=float)
    LAT, LON = np.meshgrid(dst_lat, dst_lon, indexing="ij")
    pts = np.stack([LAT.ravel(), LON.ravel()], axis=-1)
    out = interp(pts).reshape(dst_lat.size, dst_lon.size)
    return out
