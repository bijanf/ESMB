# Changelog

## [Unreleased] - 2026-06-14b — Coupled stability verified + cold-SST-drift fix

### Coupled stability — STABLE (3-year run, adversarially audited)
A 3-year COUPLED run (`DinoAtmosphere` <-> ocean, daily coupling) stayed finite and bounded
throughout — every metric-log row and all 6 snapshots `finite=True`, no NaN, no runaway, no
clamp/clip-pinning. A 5-agent audit (ocean T/S, atmosphere, conservation, AMOC) returned
STABLE-WITH-BIASES: volume-mean salinity conserved to ~0.4%, atmosphere winds equilibrated
(|u| 61.7 +/- 5.3 m/s, trending mildly *down*), AMOC finite (+4.6 Sv upper / -3.3 Sv lower at
26.5N, correct two-cell sign). The atmosphere alone is also stable (400-day SST-forced run,
EKE equilibrates ~day 120, |u| bounded 47-83). The default coupled atmosphere is still the
single-level model; `DinoAtmosphere` remains experimental.

### Cold-SST-drift fix (`experiments/run_dino_coupled.py` `ocean_fluxes`)
The audit found a cold-SST equilibrium bias (SST drifted 17.3 -> ~12.6 C over a few years,
decelerating). CAUSE: the heat-flux adjustment removed the global-mean net heat over ALL grid
cells (land+ocean), but `step_ocean` applies heat only to OCEAN cells; the land cells carry an
unphysical `net_heat` from land "SST" values, so the all-cell mean left a residual net OCEAN
cooling. FIX: balance the heat-flux adjustment over OCEAN cells only (new `ocean_mask` arg).
VERIFIED by a 1.5-year re-run: the SST drift `17.29 -> 14.87 C` (-2.42, old) becomes
`17.29 -> 17.50 C` (+0.21, flat); the polar over-cooling is also alleviated (`Tmin -4.7 -> -1.3 C`).
Still bounded/finite. Remaining biases (deferred): no ocean freezing floor; weak/noisy AMOC;
AMOC streamfunction diagnostic artifact (spurious tropical extrema from barotropic-throughflow
removal in the coarse open-box Atlantic).

## [Unreleased] - 2026-06-14 — Multi-level atmosphere: eddy-driven surface westerlies

The dinosaur atmosphere had a realistic ZONAL-MEAN circulation but its surface
DYNAMIC fields (`u_sfc`/`v_sfc`/`mslp`) had ~0 pattern correlation with ERA5.
Root cause found and fixed.

### Root cause — the circulation was perfectly AXISYMMETRIC (eddy-free)
- `isothermal_rest_atmosphere` initializes an EXACTLY zonally-symmetric state
  (vorticity == divergence == temperature_variation == 0). That is an *unstable*
  equilibrium the flow never leaves on its own — only floating-point roundoff
  breaks the symmetry, after ~100 days. With no baroclinic eddies there is no eddy
  momentum-flux convergence, hence NO mid-latitude surface westerlies: the surface
  stayed easterly at every latitude, so the `u_sfc` pattern correlation was ~0.
- Two things made the eddy onset slow even once seeded: (a) the slow (40-day)
  Held-Suarez thermal relaxation means the baroclinic jet takes ~50-60 days to
  build from isothermal rest; (b) the strong `tau=2 h` hyperdiffusion damps the
  high-wavenumber (l~10-15) eddies that carry the momentum flux at coarse T31 on
  1-15 days — competitive with their ~1-3 day baroclinic growth — suppressing them.
- Diagnosed by tracking eddy kinetic energy + the surface-U latitude profile: with
  the old setup EKE stayed *exactly* 0 and the profile was perfectly hemispherically
  symmetric; an injected velocity perturbation decayed; the dry Held-Suarez control
  had the SAME defect (it had only a thermal-wind jet, never an eddying state).

### Fix (`chronos_esm/atmos/dino_atmos.py`)
- **Near-equilibrium initialization**: `initial_state(sst)` sets the initial
  temperature to the SST-anchored radiative-equilibrium `Teq` (not isothermal rest),
  so the equator-pole gradient and the baroclinic jet exist from day 0 instead of
  after ~2 months of relaxation.
- **Rotational symmetry-breaking seed**: a small random VORTICITY perturbation is
  baked into the base state. A temperature seed does NOT work (the thermal
  relaxation just damps it); a velocity seed projects onto the growing baroclinic
  mode and reliably triggers eddies.
- **Weaker hyperdiffusion** (`diffusion_tau_hours` 2 -> 6): lets the T31 baroclinic
  eddies grow while still killing grid-scale noise (the top mode still e-folds in
  `tau`). This is the OPPOSITE of the single-level model's "keep diffusion strong"
  rule — there eddies were spurious; here they are the physics we want.
- Callers (`benchmark_dino.py`, `run_dino_coupled.py`, `tests/test_dino_atmos.py`)
  pass the SST to `initial_state()`.

### Result
Eddies develop within ~2-6 weeks instead of never; the atmosphere now performs
genuine baroclinic instability (transient storm-track eddies, EKE ~70-200 m^2/s^2,
finite/stable). At a realistic equator-pole gradient (idealized 45 K aquaplanet) it
EARNS the textbook eddy-driven surface tripole: easterly trades, mid-latitude SURFACE
WESTERLIES (+3-4 m/s at ~50 deg), polar easterlies.

HONEST CAVEAT — the real WOA SST gradient at T31 is only ~22.6 K equator-pole (T31
smooths out the sharp Gulf Stream / Kuroshio / ACC SST fronts), about half the
idealized 45 K. At this weak gradient the eddy momentum flux is too small to build a
clean mid-latitude surface-westerly belt (a 22.6 K AQUAPLANET stays surface-easterly
at midlat too — it is the gradient, not the orography), so the benchmark surface
DYNAMIC-field PATTERN correlations are essentially unchanged: `u_sfc` corr -0.04 ->
-0.01, `v_sfc` -0.13, `mslp` -0.19. Those also need realistic stationary-wave forcing
(monsoons, land contrast) that the HS + bulk-moisture physics does not have. The fix
DID help the THERMODYNAMICS / hydrological cycle: `precip` corr 0.16 -> **0.34**
(std-ratio 0.80), `t2m` corr 0.67. (Benchmark: `--spinup 150 --avg 90 --sample-every 2`
on WOA SST; the dense sampling is needed because the vigorous transient eddies
otherwise swamp the weak time-mean in a short climatology.)

Net: a real correctness improvement (the atmosphere had been running an unphysical
eddy-free AXISYMMETRIC circulation) plus better precip/ITCZ — but the surface
wind/pressure PATTERN skill at T31 remains forcing/physics-limited. Recovering observed
surface westerlies would need higher resolution (to resolve SST fronts) and/or an
explicit eddy-momentum / surface-stress parameterization (as the single-level model
uses to reach `u_sfc` corr ~0.72).

## [Unreleased] - 2026-06-11g — Multi-level atmosphere: ETOPO orography

Give the dinosaur atmosphere real topography (it was aquaplanet), for stationary
waves and a topographically-shaped surface pressure.

### Added / Changed (chronos_esm/atmos/dino_atmos.py)
- **ETOPO orography** passed to the dinosaur init (`surface_height`) and the
  dycore, interpolated onto the Gaussian grid index-preserving in longitude (same
  convention as the coupled SST regrid, so SST and orography stay aligned).
- **Orography smoothing** (mild Gaussian): raw ETOPO at T31 has sharp gradients
  that produce Gibbs ripples in the spectral representation and a spurious mass
  leak (surface pressure drifted ~1.8 hPa/day). Smoothing cut it ~6x.
- **Dry-mass fixer**: each interval, rescale surface pressure to the initial
  area-weighted global mean -> global-mean pressure now perfectly stable (0 drift
  over 40+ days), fixing the MSLP bias.
- `diagnostics()` reduces surface pressure to MEAN SEA LEVEL with a fixed
  standard-atmosphere scale height (`mslp`); benchmark uses it.

### Benchmark (90-day spin-up vs single-level baseline; with orography)
- `mslp`: bias fixed by the mass fixer (the unbalanced version drifted to -145 hPa;
  now -11.7 hPa). `precip`: corr 0.16 -> **0.33**, std-ratio 0.41 -> 0.83, bias
  -1.82 -> **-0.54** (real ITCZ). `t2m`: bias 2.42 -> -0.76, std-ratio 1.53 -> 0.71.
  `sss`: 0.84 -> 0.88. Orography also lifted the surface-wind correlations slightly.
- **Still weak**: the surface DYNAMIC-field *pattern* correlations (u_sfc ~0,
  v_sfc/mslp negative). The model's zonal-mean circulation is realistic but its 2-D
  stationary-wave / surface-wind pattern does not yet match ERA5 -- the next tuning
  frontier (surface/boundary-layer winds, stationary-wave forcing, seasonal cycle,
  fuller spin-up), before making dinosaur the default coupled atmosphere.

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
