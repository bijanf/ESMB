"""
Area-weighted skill metrics for model-vs-observation comparison.

All functions are pure NumPy and operate on 2-D (nlat, nlon) fields that are
already on the same grid. NaNs (e.g. land in an ocean field) are ignored: a
cell only contributes where BOTH model and obs are finite (and the optional
mask is True). Weights are area weights (typically cos(lat)) broadcast to the
field shape.
"""

import numpy as np


def _prep(model, obs, weights, mask):
    """Broadcast inputs and build the weight array, zeroing invalid cells."""
    model = np.asarray(model, dtype=float)
    obs = np.asarray(obs, dtype=float)
    w = np.broadcast_to(np.asarray(weights, dtype=float), model.shape).astype(float).copy()

    valid = np.isfinite(model) & np.isfinite(obs) & np.isfinite(w)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    w[~valid] = 0.0
    return model, obs, w, valid


def _wmean(x, w):
    sw = w.sum()
    if sw <= 0:
        return np.nan
    return float(np.sum(np.where(w > 0, x, 0.0) * w) / sw)


def global_mean(field, weights, mask=None):
    """Area-weighted global mean of a single field (NaNs ignored)."""
    field, _, w, _ = _prep(field, field, weights, mask)
    return _wmean(field, w)


def area_weighted_stats(model, obs, weights, mask=None):
    """Full set of area-weighted skill metrics for one variable.

    Returns a dict with:
      bias        : weighted mean(model) - mean(obs)
      rmse        : weighted root-mean-square error (uncentered)
      crmse       : centered RMSE (anomaly RMSE, the Taylor-diagram radius)
      corr        : weighted spatial (pattern) correlation
      std_model   : weighted spatial std of model
      std_obs     : weighted spatial std of obs
      mean_model  : weighted mean of model
      mean_obs    : weighted mean of obs
      n           : number of valid cells
    """
    model, obs, w, valid = _prep(model, obs, weights, mask)
    n = int(valid.sum())
    if n == 0:
        nan = float("nan")
        return dict(bias=nan, rmse=nan, crmse=nan, corr=nan, std_model=nan,
                    std_obs=nan, mean_model=nan, mean_obs=nan, n=0)

    mbar = _wmean(model, w)
    obar = _wmean(obs, w)
    bias = mbar - obar
    rmse = np.sqrt(_wmean((model - obs) ** 2, w))

    mp = model - mbar
    op = obs - obar
    std_m = np.sqrt(_wmean(mp ** 2, w))
    std_o = np.sqrt(_wmean(op ** 2, w))
    cov = _wmean(mp * op, w)
    corr = cov / (std_m * std_o) if (std_m > 0 and std_o > 0) else np.nan
    crmse = np.sqrt(_wmean((mp - op) ** 2, w))

    return dict(bias=bias, rmse=rmse, crmse=crmse, corr=corr,
                std_model=std_m, std_obs=std_o,
                mean_model=mbar, mean_obs=obar, n=n)


def zonal_mean(field, mask=None):
    """Mean over longitude (axis=-1), ignoring NaN/masked cells.

    Returns a 1-D array of length nlat. Pure spatial average per latitude
    band (longitude cells are treated as equal-area within a band).
    """
    field = np.asarray(field, dtype=float)
    valid = np.isfinite(field)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    fld = np.where(valid, field, 0.0)
    count = valid.sum(axis=-1)
    summ = fld.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, summ / np.maximum(count, 1), np.nan)
    return out


def taylor_stats(model, obs, weights, mask=None):
    """Normalized stats for a Taylor diagram: (corr, std_ratio, crmse_ratio).

    std_ratio   = std_model / std_obs
    crmse_ratio = centered RMSE / std_obs
    A perfect model sits at corr=1, std_ratio=1, crmse_ratio=0.
    """
    s = area_weighted_stats(model, obs, weights, mask)
    std_o = s["std_obs"]
    ratio = s["std_model"] / std_o if std_o > 0 else np.nan
    crmse_ratio = s["crmse"] / std_o if std_o > 0 else np.nan
    return dict(corr=s["corr"], std_ratio=ratio, crmse_ratio=crmse_ratio,
                std_obs=std_o)
