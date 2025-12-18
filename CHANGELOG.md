# Changelog

## [Unreleased] - 2025-12-18

### Fixed
- **Southern Ocean Artifacts**: Fixed "crazy red boxes" in coupling difference maps by correctly masking sea ice regions where liquid ocean temperature differs from air temperature.
- **High AMOC**: Fixed unrealistic AMOC values (~3000 Sv) by implementing Rayleigh Friction ($r=2 \times 10^{-5} s^{-1}$) in the ocean geostrophic solver (`veros_driver.py`).
- **Model Stability**: Added a "Circuit Breaker" in `main.py` to abort the simulation (force NaNs) if ocean velocities exceed 5.0 m/s.
- **Diagnostics**: 
    - Updated `plot_production.py` to calculate 30-year climatology (manual loop fallback).
    - Added AMOC time series plotting.

### Added
- `debug_southern.py`: Diagnostic script for Southern Ocean artifacts.
- `analyze_instability.py`: Script to analyze polar instability and equatorial artifacts.
- `tests/test_friction.py`: Test script for verifying friction implementation.

### Known Issues
- **Polar Instability**: Occasional extreme air temperature spikes (>50°C) observed at the South Pole (Lat Index 3), though transient.
- **Equatorial Artifacts**: Some high-difference coupling pixels exist near the equator, likely due to coastal masking.
