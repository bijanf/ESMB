# Changelog

## [Unreleased] - 2026-06-11f — Multi-level atmosphere: benchmark + io (Phase 4)

### Added
- **`chronos_esm/atmos/dino_atmos.save_state/load_state`**: persist the dinosaur
  modal State (complex spectral coefficients + humidity tracer + sim_time) to .npz
  for restarting long coupled runs. Round-trip verified.
- **`experiments/benchmark_dino.py`**: spins the SST-coupled atmosphere up on the
  WOA SST and scores the climatology vs ERA5/WOA18 with the existing scorecard.

### Benchmark (90-day spin-up vs single-level baseline; honest, mixed)
Multi-level WINS on the fields driven by real thermodynamics/physics:
- `t2m`: bias 2.42 -> 0.01 K, std-ratio 1.53 -> 0.68 (warm bias gone).
- `precip`: corr 0.16 -> 0.24, std-ratio 0.41 -> 0.87, and a REAL ITCZ from the
  resolved Hadley circulation (the single-level had none); global mean ~2.6 mm/day.
- `sss`: corr 0.84 -> 0.88.

Multi-level is WORSE on two fields, for understood reasons:
- `u_sfc`: corr 0.72 -> 0.04. The single-level number was PRESCRIBED by the
  wind-relaxation parameterization; the multi-level earns its winds dynamically.
  The low 2-D correlation is mostly the aquaplanet's lack of zonal asymmetry +
  an over-damped bottom level + incomplete spin-up.
- `mslp`: corr 0.07 -> -0.25. The dinosaur atmosphere is currently AQUAPLANET (no
  orography), so it cannot reproduce ERA5's topography-dominated sea-level-pressure
  pattern / stationary waves.

### Remaining to make the multi-level uniformly better
ETOPO orography in the dycore (mslp + stationary waves); revisit the surface-wind
level / boundary-layer drag; a longer spin-up to full equilibrium; then wire the
dinosaur atmosphere into the main coupler + dashboard as the default atmosphere.

## [Unreleased] - 2026-06-11e — Multi-level atmosphere: prognostic moisture (Phase 3a)

Add a hydrological cycle to the dinosaur multi-level atmosphere
(`chronos_esm/atmos/dino_atmos.py`).

### Added
- **Prognostic specific humidity** carried as a dinosaur tracer (advected by the
  dycore), with a surface **evaporation** source (bulk formula from SST) and a
  **large-scale condensation** sink that rains out supersaturation and latent-heats
  the column. `diagnostics()` now returns `specific_humidity` and a column
  `precip` field [kg/m^2/s]. All physics is done in plain jax with float scale
  factors (no pint inside jit). Updated `tests/test_dino_atmos.py` to check the
  moisture cycle.

### Result
- **Precipitation magnitude fixed**: global-mean precip ~2.6 mm/day (observed
  ~2.9), vs the single-level model's ~1.1 mm/day dry bias. Surface humidity builds
  to a realistic ~9 g/kg. Stable and finite.
- **ITCZ emerges with equilibration**: as the Hadley cell spins up (jet -> ~13 m/s
  by day 60), tropical precip increasingly dominates (trop/subtrop ratio 0.95 ->
  1.46) and the subtropics/poles dry out -- a real ITCZ forming from the resolved
  circulation, with no convection scheme (the single-level model could not do this
  at all). It sharpens further as the jet approaches its ~25 m/s equilibrium.

### Phase 3b: consistent moist coupling + energy balance (run_dino_coupled.py)
- The coupled ocean fluxes now use the atmosphere's OWN near-surface humidity and
  precipitation: latent heat / evaporation from `q_sfc`, freshwater = real P - E.
- **Heat-flux adjustment**: remove the area-weighted global-mean net heat so the
  control run has no net global ocean heating/cooling (cuts the SST drift roughly
  in half: ~-0.27 C / 30 days). Coupled run is stable with realistic winds
  (|u|max ~16-19 m/s) and wind-driven currents (~0.06 m/s).
- Remaining (Phase 4): persist the dinosaur modal state in io.py; long coupled
  spin-up + benchmark the multi-level atmosphere vs ERA5 in the dashboard.

## [Unreleased] - 2026-06-11d — Multi-level atmosphere: dinosaur dycore (Phase 1)

Begin replacing the single-level (barotropic) atmosphere — which fundamentally
cannot produce synoptic systems or a real ITCZ — with a MULTI-LEVEL spectral
primitive-equation dycore, to fix the limitations documented in the v3 overhaul.

### Added
- **`dinosaur-dycore==1.2.1`** dependency: Google Research's differentiable JAX
  spectral primitive-equation dycore (the engine behind NeuralGCM). Compatible
  with jax 0.8.1 — no jax upgrade; the existing model and test suite still pass.
- **`experiments/dino_held_suarez.py`**: validated reference + reusable builder
  (`build_held_suarez_model`). Runs dinosaur's dry dycore (T31, 24 sigma levels,
  ImEx-RK3 + del^4 hyperdiffusion filter) with Held-Suarez (1994) forcing and
  reproduces the textbook baroclinic circulation from an isothermal rest state:
  twin ~22 m/s upper-tropospheric mid-latitude westerly jets + surface trade
  easterlies (`docs/figures/dino_held_suarez_jet.png`). This is the circulation
  the single-level model could never generate — proof the dycore is the right tool.

### Added (Phase 2a: SST-coupled atmosphere module)
- **`chronos_esm/atmos/dino_atmos.py`** (`DinoAtmosphere`): wraps dinosaur's dry
  dycore with an SST-COUPLED thermal forcing. The radiative-equilibrium
  temperature of a Held-Suarez-style relaxation is anchored to the underlying SST
  at the surface (instead of HS's idealized 315 - 60 sin^2(lat)), keeping HS's
  vertical lapse, boundary-layer relaxation rates, and Rayleigh drag -- so the
  atmosphere RESPONDS to the ocean. SST is constant within an ocean coupling
  interval, so `run_interval(state, sst, n_steps)` is jitted once with SST traced.
- **`tests/test_dino_atmos.py`**: validated that a realistic SST gradient drives a
  baroclinic mid-latitude jet (both hemispheres), surface trade easterlies, and a
  lower atmosphere that tracks the SST gradient (>10 K equator-pole), staying finite.

### Added (Phase 2b: coupled dinosaur atmosphere <-> ocean)
- **`experiments/run_dino_coupled.py`**: a stable coupled run of the multi-level
  dinosaur atmosphere with the Chronos ocean. Sequential coupling at a fixed
  interval: ocean SST -> (regrid linear->Gaussian) -> `DinoAtmosphere.step` ->
  surface winds/T -> (regrid back) -> bulk fluxes (wind stress + SW/LW/sensible/
  latent net heat) -> `step_ocean`. Includes a 1-D Gaussian<->linear latitude
  regridder (longitudes coincide). Validated 30 days: the ocean SST gradient
  drives a baroclinic mid-latitude jet (->+6 m/s, still spinning up), the winds
  drive ocean currents (~0.04 m/s), SST stays stable (~17 C, slow drift), all
  finite. Moisture is a fixed-RH boundary-layer closure for surface latent heat;
  prognostic moisture/precip (ITCZ) and flux balancing are Phase 3.

### Notes
- **`jcm` is NOT usable** and `chronos_esm/atmos/jcm_adapter.py` is dead: jcm 1.1.1
  on PyPI is built against an unreleased dinosaur (imports `SI_SCALE`, which exists
  in no released dinosaur) and runs inert. We build on `dinosaur` directly instead.
- Next (Phases 3-4): prognostic moisture + large-scale condensation for the ITCZ;
  balance the surface energy budget (remove the slow SST drift); persist the modal
  state in io.py; extract surface fields (u10/t2m/mslp/precip) and benchmark the
  multi-level atmosphere vs ERA5 in the dashboard.

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
