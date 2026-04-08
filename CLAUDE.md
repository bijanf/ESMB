# Chronos-ESM Developer Context

## Project Overview
**Chronos-ESM** is a JAX-based, fully differentiable Earth System Model (Ocean, Atmos, Land) running on GPU.
*   **Resolution**: T31 (approx 3.75°).
*   **Physics**: Primitive equations (Ocean), QTCM (Atmos).
*   **State**: Physics overhaul complete, ready for validation runs.

## Current State: Physics Overhaul (v2)

The model has been upgraded from "tank mode" (extreme artificial damping) to physically-grounded dynamics. Six phases were implemented:

### Phase 1: Semi-Implicit Atmospheric Time Stepping - DONE
*   **File**: `chronos_esm/atmos/spectral.py` - `solve_helmholtz()` function
*   **File**: `chronos_esm/atmos/dynamics.py` - Lines 606-673
*   Gravity waves treated implicitly via Helmholtz equation
*   Linearized around T0=250K reference temperature
*   Energy fixer alpha reduced from 0.1 to 0.01 (semi-implicit is nearly conservative)

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

### Phase 4: Reduced Artificial Damping - DONE
*   **File**: `chronos_esm/atmos/dynamics.py` - Hyperdiffusion (lines 590-597)
*   Hyperdiffusion reduced ~4x: `nu4_vort=8e13, nu4_div=1.5e14, nu4_temp=5e12`
*   Wind caps raised from 80 → 150 m/s (allow jet stream dynamics)
*   Vorticity clamp U_MAX raised from 80 → 150 m/s

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
