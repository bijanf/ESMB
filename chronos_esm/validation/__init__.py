"""
Validation framework for Chronos-ESM.

Quantitative benchmarking of model output against observations / reanalysis:
  - obs.py       : loaders for WOA18 (ocean T/S) and ERA5 (atmosphere) + a few scalar benchmarks
  - grid.py      : model grid helpers + regridding of obs onto the model grid
  - metrics.py   : area-weighted skill metrics (bias, RMSE, pattern corr, Taylor stats, zonal means)
  - plots.py     : Nature-style figures (bias maps, zonal-mean overlays, Taylor diagram, drift series)
  - scorecard.py : orchestrates obs-vs-model scoring into a printable scorecard

This package only *reads* model output; it never mutates model state.
"""

from chronos_esm.validation import grid, metrics, obs, plots, scorecard  # noqa: F401
