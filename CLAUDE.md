# Chronos-ESM Developer Context

## Project Overview
**Chronos-ESM** is a JAX-based, fully differentiable Earth System Model (Ocean, Atmos, Land) running on GPU.
*   **Resolution**: T31 (approx 3.75°).
*   **Physics**: Primitive equations (Ocean), QTCM (Atmos).
*   **State**: 100-Year Control Run in progress.

## Current Mission: Century Run
We are attempting to run a 100-year control simulation.
*   **Progress**: Years 0-42 are **Verified Stable**.
*   **Bottleneck**: Years 42-47 were **Unstable** (now fixed - see below).
    *   Root cause: CFL violation at poles + Helmholtz solver blow-up where dx→0.
    *   The model crashed around Year 39 or Year 47 due to **Supersonic Winds (>160 m/s)** near the South Pole.

## Root Cause Analysis (Completed)
The polar instability was caused by:
1.  **CFL Violation**: Grid spacing `dx = 2πR·cos(lat)/nlon → 0` at poles, violating stability.
2.  **Helmholtz Blow-up**: Tridiagonal solver has wavenumber `kx ∝ 1/dx → ∞` at poles.
3.  **Late Polar Filtering**: Filter was applied AFTER the explosive growth in the solver.

## Fixes Applied (Attempt 10)

### Fix 1: Pre-Helmholtz Polar Filtering
*   **File**: `chronos_esm/atmos/dynamics.py` (lines 219-227)
*   **Change**: Apply polar filter to vorticity/divergence BEFORE inverse Laplacian.
*   **Effect**: Prevents high-k modes from triggering Helmholtz resonance at poles.

### Fix 2: Latitude-Dependent Diffusion
*   **File**: `chronos_esm/atmos/dynamics.py` (lines 379-388)
*   **Change**: Diffusion coefficient `nu` now scales with `1/cos(lat)²` at high latitudes.
*   **Effect**: Stronger damping where CFL is violated, up to 25x at poles.

### Fix 3: Safe Resume Point
*   **File**: `experiments/run_century_resume_tank.py`
*   **Change**: Resume from `year_042.nc` (verified safe), not `year_046.nc` (poisoned).

## Strategy: "Stabilized Tank Mode"
*   **Viscosity (Ah)**: `5.0e6` (maximum ocean viscosity).
*   **Timestep (DT_ATMOS)**: `30s` (minimum safe timestep).
*   **Shapiro Filter**: `0.9` (strong smoothing).
*   **Pre-Helm Filter**: **Enabled** (new fix).
*   **Polar Diffusion**: **25x boost** at high latitudes (new fix).

## Resume Points (In `outputs/century_run/`)
*   `year_042.nc`: **SAFE**. The last known "clean" state (Low noise, stable energy).
*   `year_046.nc`, `year_047.nc`: **POISONED**. Contains hidden instability. **Do not use.**

## Key Scripts
*   **`experiments/run_century_resume_tank.py`**:
    *   The current "Rescue Script" (Attempt 10).
    *   Configured for **Ah=5e6**, **DT=30s**, resumes from **Year 42**.
    *   Includes enhanced stability monitoring (wind, vorticity tracking).
*   **`experiments/check_production_stability.py`**:
    *   Scans output files for NaN and Stability Metrics (Global T, AMOC, Max Wind).
*   **`experiments/find_hotspots.py`**:
    *   Diagnoses individual year files for instability indicators.
*   **`chronos_esm/config.py`**:
    *   Global physics constants.

## How to Resume
1.  Verify `year_042.nc` is healthy:
    ```bash
    python experiments/find_hotspots.py outputs/century_run/year_042.nc
    # Should show: Max Wind < 50 m/s, Grid Noise < 2.0
    ```
2.  Submit via SLURM:
    ```bash
    sbatch experiments/run_century_resume_tank_slurm.sh
    ```
3.  Monitor logs:
    ```bash
    tail -f logs/resume_tank_*.log
    python experiments/check_production_stability.py
    ```

## Known Issues
*   **AMOC Scaling**: The raw AMOC calculation requires a 0.15 depth factor for the surface layer to report realistic Sverdrups (~18 Sv).
*   **Polar Winds**: If `Atmos U` exceeds 80 m/s, monitor closely. Above 120 m/s triggers emergency checkpoint.

## Status Codes (from `check_production_stability.py`)
*   **OK**: Running normally.
*   **UNSTABLE (Wind)**: Max wind > 80 m/s. Monitor closely.
*   **CRASH (NaN)**: Simulation blew up.

## Technical Details: The Polar Instability

The instability chain was:
1. Baroclinic eddies develop polar vorticity anomalies (normal physics).
2. When vorticity passes to Helmholtz solver with `dx ∝ cos(lat) → 0`:
   - Wavenumber `kx = 2πk/(nlon·dx) → ∞`
   - Tridiagonal matrix becomes ill-conditioned
   - Solution amplifies high-frequency noise
3. Wind recovery `u = -∂ψ/∂y + ∂χ/∂x` amplifies the noise further.
4. Advection with `u·∂/∂x` where `∂/∂x ∝ 1/dx` creates explosive tendencies.
5. After ~7 days of error accumulation, winds exceed physical bounds → NaN.

The fix targets step 2 (filter input to prevent resonance) and step 4 (boost diffusion where dx is small).

*Last Update: Attempt 10 with Pre-Helmholtz filter + Polar diffusion boost.*
