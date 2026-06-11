# Chronos-ESM Developer Context

## Project Overview
**Chronos-ESM** is a JAX-based, fully differentiable Earth System Model (Ocean, Atmos, Land) running on GPU.
*   **Resolution**: T31 (approx 3.75°).
*   **Physics**: z-level primitive equations (Ocean); single-level spectral
    primitive-equation dynamics (Atmos, `atmos/dynamics.py` — barotropic, forward
    Euler). A `qtcm.py` and a dinosaur/JCM adapter (`jcm_adapter.py`) also exist but
    are NOT the active atmosphere path.
*   **State**: Ocean + atmosphere correctness passes done and benchmarked vs
    WOA18+ERA5 (see Validation dashboard in README). Single-level atmosphere limits
    remain (no synoptic systems, weak ITCZ).

## Current State: Atmosphere correctness overhaul (v3, 2026-06-11)

The atmosphere was the weak link in the validation dashboard. Diagnosed + fixed
the root causes (see CHANGELOG `2026-06-11c`). Benchmark (30-day WOA spin-up vs
WOA18+ERA5): `mslp` std-ratio 14.67 -> 2.07, bias -71.8 -> +2.9 hPa; `u_sfc` corr
-0.08 -> 0.72; `precip` corr -0.06 -> 0.16; sst/sss held. Key facts for future work:
*   The atmosphere is **single-level (barotropic), forward-Euler** (`atmos/dynamics.py
    step_atmos`). It has NO baroclinic eddies -> no synoptic systems (mslp/v_sfc
    pattern corr ~0) and a weak ITCZ (precip too dry, rains in subtropics not at the
    equator). Closing these needs vertical structure (multi-level or external dycore).
*   **MSLP blow-up was a POLAR instability**: the dynamical `dx` collapsed at the
    poles; `advect(ln_ps)` + the `R*T*lap(ln_ps)` term detonated there. Fixed by
    flooring the dynamical `dx` at its 80-deg value (`cos_lat_dyn`).
*   **Winds are relaxed toward a climatological zonal jet** (`u_target`, tau=2 days),
    NOT toward rest — a barotropic single level can't generate jets itself. This is
    an eddy-momentum-flux parameterization; tune `u_target` / tau there.
*   **`phi_s` is now persisted in checkpoints** (`io.py`); older checkpoints get it
    reconstructed from ETOPO on load. Restarts no longer lose topography.

### Phase 1: Semi-Implicit Atmospheric Time Stepping - NOT WIRED IN (aspirational)
*   `solve_helmholtz()` exists in `chronos_esm/atmos/spectral.py` but is **unused** —
    `step_atmos` is forward Euler. The "lines 606-673" referenced here never existed.
*   At dt=30 s the gravity-wave CFL is tiny (~0.06), so forward Euler's per-step
    growth is negligible; semi-implicit was NOT the fix the dashboard needed (the
    pole-`dx` collapse was). Wire in `solve_helmholtz` only if dt is raised.
*   Energy fixer alpha IS reduced 0.1 -> 0.01 (done, in `step_atmos`).

### Phase 2: Dynamic Wind Stress - DONE
*   **File**: `chronos_esm/main.py` - Lines 300-333
*   Wind stress computed from atmospheric surface winds: `tau = rho_air * C_d * |V| * V`
*   Gustiness floor of 1.0 m/s, clamped to ±0.5 Pa
*   Blends from prescribed to dynamic over first 5 years

### Phase 3: Prognostic Salinity + Freshwater - DONE
*   **File**: `chronos_esm/ocean/veros_driver.py` - Salinity sponge (lines 173-185)
*   **File**: `chronos_esm/main.py` - River runoff routing (lines 237-255)
*   Global 35 psu relaxation replaced with high-lat sponge (>70°, 6-month timescale)
*   Salinity clamp widened: [25,45] → [20,42]
*   Land runoff routed to adjacent coastal ocean cells

### Phase 4: Atmospheric damping - REVERTED to strong (see v3 overhaul)
*   **File**: `chronos_esm/atmos/dynamics.py` - Hyperdiffusion block
*   Hyperdiffusion is back at the strong "tank" values `nu4_vort=3e14,
    nu4_div=6e14, nu4_temp=2e13`. With the v3 wind relaxation now SETTING the jet,
    dissipation no longer has to be weak to "let the jet develop" — del^4 only hits
    small scales and doesn't fight the large-scale relaxed jet. Keeping it strong is
    what makes the run stable: a 4x-weaker setting let a slow eddy instability grow
    (|u| -> clamp by ~day 25). This DECOUPLING (relaxation -> realism, hyperdiffusion
    -> stability) is the key idea — do not weaken hyperdiffusion to chase winds.
*   Wind clamp is **80 m/s** (lowered from 100) — a tight safety net; the relaxed
    field sits near ~10 m/s so hitting it flags an instability.

### Phase 5: AMOC Diagnostics - DONE
*   **File**: `chronos_esm/ocean/diagnostics.py`
*   `create_atlantic_mask()` - Atlantic basin (280-360E + 0-20E, 34S-80N)
*   `compute_amoc()` - Atlantic overturning streamfunction + 26.5N metrics
*   `compute_amoc_diagnostics()` - Scalar metrics for logging

### Phase 6: Experiment Script - DONE
*   **File**: `experiments/run_century_physics.py` - Main run script
*   **File**: `experiments/run_century_physics_slurm.sh` - SLURM submission
*   Physically realistic parameters (Ah=5e4, shapiro=0.1, kappa_gm=1000, kappa_h=500)
*   AMOC diagnostics every 500 steps, streamfunction every 5 years
*   Resume capability from any checkpoint

## Experiment Parameters Comparison

| Parameter | Tank Mode (old) | Physics Mode (new) |
|-----------|----------------|-------------------|
| Ah (ocean viscosity) | 5.0e6 | 5.0e4 |
| shapiro_strength | 0.9 | 0.1 |
| kappa_gm | 3000 | 1000 |
| kappa_h | 2000 | 500 |
| nu4_vort | 3.0e14 | 8.0e13 |
| nu4_div | 6.0e14 | 1.5e14 |
| nu4_temp | 2.0e13 | 5.0e12 |
| Wind caps | 80 m/s | 150 m/s |
| Energy fixer alpha | 0.1 | 0.01 |
| Wind stress | Prescribed | Dynamic (bulk drag) |
| Salinity | Global relax to 35 | High-lat sponge only |
| Time stepping | Forward Euler | Semi-implicit |

## How to Run (New Physics)

### Fresh start:
```bash
sbatch experiments/run_century_physics_slurm.sh
```

### Resume from checkpoint:
```bash
sbatch experiments/run_century_physics_slurm.sh --resume year_042
```

### Monitor:
```bash
tail -f logs/century_v2_*.log
```

## Verification Targets

After running:
1. **5-year smoke test** from fresh initialization — no NaN, no crash
2. **Energy drift** < 0.5%/year without energy fixer (or with alpha=0.01)
3. **AMOC** at 26.5N in range 5-25 Sv (observed: ~17 Sv)
4. **Trade winds** consistently easterly at 10-20° latitude (-5 to -15 m/s)
5. **Global mean temperature** stable at 285±3K
6. **Salinity** global mean stable at 34.5-35.5 psu over 10 years
7. **Jet stream** winds reach 100+ m/s without clamping intervention

## Legacy Resume Points (In `outputs/century_run/`)
*   `year_042.nc`: Verified clean state (from tank mode).
*   Can be used as resume point for new physics mode.

## Key Scripts (Legacy)
*   **`experiments/run_century_resume_tank.py`**: Old tank mode script.
*   **`experiments/run_century_resume_tank_slurm.sh`**: Old SLURM submission.

## Known Issues (Pre-Overhaul, May Be Resolved)
*   **Year 48 Instability**: Caused by Forward Euler phase errors on gravity waves.
    *   Semi-implicit time stepping (Phase 1) should resolve this root cause.
*   **Vorticity Clamp**: Was hitting 80 m/s limit — now raised to 150 m/s.

*Last Update: Physics overhaul complete (Phases 1-6). Ready for validation.*
