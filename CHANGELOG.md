# Changelog

## [Unreleased] - 2026-06-11c — Atmosphere correctness overhaul

The validation dashboard flagged the atmosphere as the weak link (sea-level
pressure variance ~15x observed, surface winds *anti*-correlated with ERA5,
precipitation pattern uncorrelated). Diagnosed each failure with instrumented
short runs, then fixed the root causes. Benchmark vs WOA18+ERA5 (30-day WOA
spin-up, mean of days 20-30): `mslp` std-ratio 14.67 -> 2.07 and bias -71.8 ->
+2.9 hPa; `u_sfc` corr -0.08 -> 0.72; `precip` corr -0.06 -> 0.16; sst/sss held.

### Fixed
- **Surface-pressure (MSLP) blow-up = a POLAR instability** (`atmos/dynamics.py`).
  The bare lat-lon `dx` collapses to ~4 m at the pole rows (`cos_lat` floored to
  1e-5), so `advect(ln_ps)` and the `R*T*lap(ln_ps)` pressure-gradient term
  exploded AT the poles within ~10 steps — ps swung to 100-1200 hPa at lat +-90
  (zero topography) and pinned to the `[9.2, 11.7]` clamp, giving ps std ~150 hPa
  vs obs ~12. Only the *diffusion* operators used the CFL-safe `dx_diff`; the
  *dynamics* used the collapsing `dx`. Floored the dynamical `dx` at its 80-deg
  value too. Result: ps now physical (549-1054 hPa, std ~48 mostly topographic),
  no rail hits, stable.
- **Surface geopotential `phi_s` not persisted** (`io.py`): the loader zeroed it
  on every restart ("should be static" — but it was never saved), silently
  removing topography from resumed runs and breaking the MSL pressure reduction
  (surface pressure stays topographically low while `phi_s` reads 0). Now saved,
  and reconstructed from ETOPO for older checkpoints that predate the field.
- **Divergence damping was numerically unstable**: the explicit `-5e-2*div` term
  gave `dt*rate = 1.5 > 1`, so forward Euler flipped the divergence sign every
  step (a 2dt oscillation pumping noise into `ln_ps`). Reapplied implicitly,
  `div/(1+dt*rate)`, unconditionally stable.

### Changed
- **Winds: relax toward a climatological jet, not toward rest** (`atmos/dynamics.py`).
  The atmosphere is single-level (barotropic): with no vertical shear it cannot do
  baroclinic instability or thermal-wind balance, so the winds geostrophically
  adjusted to a near-standstill (|u| 10 -> <1 m/s, corr vs ERA5 ~0). Replaced the
  Rayleigh drag-toward-zero with a 2-day relaxation toward an observed zonal-mean
  surface-wind profile (tropical easterlies + mid-latitude westerlies + weak polar
  easterlies), parameterizing the unresolved eddy-momentum flux. 2-day (not 6) so
  the damping beats the barotropic instability of the relaxed jet's shear (at 6
  days localized eddies ran |u| to the clamp by ~day 25). Wind safety clamp
  100 -> 80 m/s. Energy-fixer alpha 0.1 -> 0.01 (scheme is near-conservative once
  the pole instability is gone). Added scale-selective del^4 smoothing of `ln_ps`.
- **MSL pressure reduction** (`validate_control.canonical_fields`): reduce surface
  pressure to sea level (`p_s * exp(Phi_s/(R_d*T_ref))`, fixed T_ref=288 K) before
  comparing to ERA5 MSL, instead of comparing raw surface pressure (which is ~550
  hPa over a 5 km plateau).
- **Dashboard climatology** (`make_readme_figures.py`): accepts a glob of states
  and averages them into a climatology (removes weather noise / spin-up transient).

### Known limitations (single-level atmosphere)
- No baroclinic eddies -> no synoptic systems: `mslp` and `v_sfc` *pattern*
  correlations stay near zero, and the **tropical ITCZ is weak**, so precipitation
  is too dry (-1.8 mm/day) and only weakly correlated (the model rains in the
  subtropics/mid-latitudes instead of at the equator). Closing these likely needs
  vertical structure (multi-level dynamics or an external dycore). A modest warm,
  over-variable near-surface air-temperature bias also remains.

## [Unreleased] - 2026-06-11b

### Fixed
- **High-latitude grid-scale SST noise** (the long-standing one): a controlled
  diagnosis (convection-off vs strong-geostrophic-drag) pinned the cause to the
  **convective adjustment**, not the geostrophic velocity — turning convection off
  cut NH SST noise ~20x while damping the velocity did nothing. Root cause: the
  hard on/off convective switch (kappa 1e-5 -> 10) fires patchily column-by-column
  and, at the thin 50 m surface layer, violates the explicit vertical-diffusion CFL
  limit. Replaced it (`ocean/mixing.py compute_vertical_diffusivity`) with a SMOOTH
  ramp of kappa with the degree of static instability plus a per-interface
  explicit-stability cap (0.45*dz^2/dt). Result (3000-step WOA run): NH SST noise
  ~50 -> 1.8, SST corr vs WOA 0.86 -> 0.97, deep convection retained, stable.
  The smooth ramp also removes the non-differentiable jnp.where step.

## [Unreleased] - 2026-06-11

### Added
- **Variable-depth bathymetry** (full-cell "staircase") derived from ETOPO
  (`data.load_ocean_depth`): depth = max(0, -elevation), block-mean coarsened,
  Gaussian-smoothed, restricted to the WOA ocean footprint, min depth 500 m,
  isolated one-cell ponds removed. `ocean/bathymetry.py` turns it into a static
  3-D wet mask + C-grid face masks (`main.ocean_masks`, `ModelParams.ocean_mask_3d`).
- **Land-aware (wet-point) flux-masked ocean operators** (`step_ocean`): tracer
  advection AND diffusion exchange only across open faces (face open iff both
  adjacent centers are wet — MITgcm hFac MIN rule), no flux through the sea floor,
  surface flux only into the top wet cell, dry cells frozen. Per-column wet depth
  `H_col` used for the baroclinic vertical-mean removal; velocities and `w` masked.
  Salt conservation now weights by the wet volume. Approach validated by deep
  research (Adcroft/Hill/Marshall 1997; MITgcm; Veros) and a 5-dimension
  adversarial code review (no defects found).

### Known Issues
- The bathymetry/flux-masking did NOT resolve the high-latitude grid-scale SST
  noise: a flat-vs-bathymetry run showed it is generated INTERNALLY in the
  high-lat dynamics (convective-adjustment patchiness / geostrophic velocity at
  small polar dx), not by land-bleed. Separate fix needed.
- Barotropic streamfunction still uses a flat reference depth (no topographic
  steering of the depth-integrated flow); variable-H external mode (∇·(1/H∇ψ),
  JEBAR) is deferred pending dedicated research.
- Partial bottom cells (hFac) deferred; full-cell staircase is the baseline.

## [Unreleased] - 2026-06-10

### Added
- **Observation-validation framework** (`chronos_esm/validation/`): scores model
  output against WOA18 (ocean T/S) and ERA5 (near-surface atmosphere) with
  area-weighted skill metrics (bias, RMSE, pattern correlation, Taylor stats),
  bias maps, zonal-mean comparisons, and a printable scorecard.
    - `experiments/validate_control.py`: score saved checkpoints or a short
      in-process demo against obs (`--era5` to include the atmosphere).
    - `experiments/make_readme_figures.py`: regenerate the README validation figures.
- **WOA-seeded ocean init**: `main.init_model(ocean_ic="woa")` and
  `run_century_physics.py --ocean-ic woa` (now default) seed ocean T/S from
  WOA18 instead of a uniform 10 °C / 35 psu ocean, drastically shortening spin-up.
- **Global water conservation**: remove the area-weighted global mean of P-E so
  global precipitation balances evaporation (no spurious freshwater source/sink).
- **Global-mean salinity conservation**: renormalise the volume-weighted ocean
  salinity to 34.7 psu each step, removing the sea-ice-brine / clip / numeric
  drift that previously ran salinity into the clip.
- **Diagnosed ocean vertical velocity** `w` from continuity (was held at zero).
- `cdsapi` dependency (ERA5) and `.gitignore` entries for pip wheels / artifacts.

### Fixed
- **Crash/NaN guards** (multi-agent code review): polar del^4/del^2 diffusion CFL
  blow-up (CFL-safe `dx_diff`), missing CO2 safety clamp, legacy-checkpoint resume
  guards (`land_temp`/`sst`/vort-div pole `dx=0`), `rotate_tensor` lost `return`,
  `CHRONOS_RESOLUTION` import crash -> warn+fallback, `run_scenario` restart
  missing `phi_s`/`soil_carbon`, `spectral` scalar-`dx` indexing, `soft_clip`
  divide-by-zero.
- **AMOC diagnostic**: use the stretched `dz`; remove the barotropic throughflow
  and restrict to ocean cells so the overturning streamfunction closes (the
  previously reported ~18-20 Sv was an open-basin artifact; 26.5N bottom value
  -201.8 -> -0.0 Sv).
- **SST/SSS collapse**: the per-step del^2 Shapiro filter (strength 0.5) erased
  resolved gradients; replaced with a scale-selective biharmonic form, off by
  default (noise control is the physical Laplacian diffusion). SST-vs-WOA18
  pattern correlation recovered from ~0.2 to ~0.95-1.0.

### Changed
- **Ocean vertical grid** stretched (50 m surface -> ~550 m deep, 15 levels,
  5000 m total) instead of uniform ~333 m; shared between dynamics `dz` and the
  IC depth coordinate. Removes the ~3x too-sluggish surface response.
- **Ocean shortwave** no longer double-counts planetary albedo (applies ocean
  albedo with atmospheric transmission), fixing a systematic cold bias.
- **River runoff** uses non-wrapping latitude shifts, so polar runoff no longer
  leaks pole-to-pole across the non-periodic boundary.
- Default ocean `shapiro_strength` 0.1 -> 0.0; README rewritten to reflect the
  validation-driven status and corrected AMOC.

### Known Issues
- **No thermohaline overturning yet** (~0 Sv): needs a multi-decade spin-up now
  that the ocean preserves realistic density structure.
- **Warm subtropical SST bias** (bounded; a shortwave/radiation calibration item).
- **Atmosphere biases**: precipitation magnitude and sea-level-pressure variance.
- **High-latitude grid-scale SST noise** (worst in the NH): land-ocean "bleed" --
  the ocean step evolves land cells and mixes them across coastlines. Naive
  del^2/del^4 filtering either collapses gradients or blows up at coasts; the
  proper fix is land-aware (no-flux at coasts) tracer advection/diffusion.

## [Unreleased] - 2025-12-23

### Added
- **Multi-Parameter Physics Tuning:**
    - Updated `chronos_esm/ocean/veros_driver.py` to accept `kappa_gm` (Gent-McWilliams diffusivity) as a dynamic argument.
    - Updated `chronos_esm/main.py` to pass `kappa_gm` through `step_coupled`.
    - Updated `experiments/tune_physics_v2.py` to optimize `r_drag` and `kappa_gm` simultaneously.
    - Added `experiments/diagnose_tuning_init.py` for debugging gradient instability.

### Fixed
- **Tuning Instability:** Fixed NaN crash at step 18 of tuning loop by implementing tighter gradient clipping and multi-parameter optimization.

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
