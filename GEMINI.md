# Chronos-ESM Context Guide

## Project Overview

**Chronos-ESM** is a fully differentiable, coupled Earth System Model implemented in **JAX**. It is designed for efficient simulation and proxy data assimilation on consumer hardware (CPU/GPU) using XLA fusion.

### Key Components
*   **Ocean:** Veros-based primitive equation model (T63 resolution: ~192x96x40). Includes a JAX-native Conjugate Gradient solver.
*   **Atmosphere:** QTCM (Quasi-Equilibrium Tropical Circulation Model) with spectral transforms.
*   **Land:** Basic land model with snow depth and temperature tracking.
*   **Coupler:** Conservative regridding and flux exchange between components.
*   **Proxy:** Bemis isotope equations for paleo-data assimilation.

## Technical Architecture

*   **Language:** Python 3.9+
*   **Core Framework:** **JAX** (v0.4+) for automatic differentiation and compilation.
*   **Data Format:** NetCDF (via `netCDF4` and `xarray`) for state serialization and analysis.
*   **Resolution:** Configurable (T31 or T63) via `chronos_esm/config.py` or environment variables.

## Directory Structure

*   **`chronos_esm/`**: Main source package.
    *   `config.py`: Global configuration (grids, physics constants, time steps).
    *   `main.py`: Core time-stepping logic.
    *   `ocean/`, `atmos/`, `land/`, `ice/`, `coupler/`: Component implementations.
    *   `proxy/`: Data assimilation layer.
*   **`experiments/`**: Scripts for configuring and running simulations.
    *   `production_control.py`: Standard control run (100 years, monthly means).
    *   `analyze_*.py`: Post-processing and plotting scripts.
    *   `run_*_slurm.sh`: SLURM submission scripts for cluster execution.
*   **`tests/`**: Unit and integration tests (run with `pytest`).
*   **`outputs/`**: Default location for simulation artifacts (NetCDF files, logs).
*   **`analysis_results/`**: Organized output for specific experiment versions.

## Development & Usage

### 1. Installation
The project relies on a standard Python environment.
```bash
pip install -r requirements.txt
```

### 2. Running Simulations
Simulations are typically defined as scripts in the `experiments/` directory.

**Critical Rule:** All simulation jobs **must** be submitted via SLURM to ensure proper resource allocation and reproducible environments. Do not run heavy python scripts directly in the login shell.

**Example: Running the Production Control Run**
```bash
sbatch experiments/run_control_slurm.sh
```
*   **Output:** `outputs/production_control/` (contains `state_0000.nc`, monthly means, and restarts).
*   **Behavior:** Runs for 100 years by default, saving monthly means and yearly restarts. Resumes automatically if restart files exist.

### 3. Configuration
Key parameters are defined in `chronos_esm/config.py`. Some can be overridden via environment variables:
*   `CHRONOS_RESOLUTION`: "T31" (default) or "T63".
*   `CHRONOS_DT_ATMOS`: Atmospheric time step.

### 4. Testing
Run the test suite to verify solver convergence and physics correctness.
```bash
pytest tests/
```

### 5. Code Style & Quality
The project enforces strict code quality standards:
*   **Formatting:** `black` and `isort`.
*   **Linting:** `flake8`.
*   **Type Checking:** `mypy`.

## Important Conventions

*   **JAX-First:** All physics logic is written using `jax.numpy` and composed via `jax.jit`.
*   **State Management:** Model state is passed as immutable NamedTuples (e.g., `OceanState`, `AtmosState`).
*   **Time Stepping:** Uses `jax.lax.scan` for efficient compilation of the time loop, especially for long integrations.
*   **IO:** Heavy I/O should be kept outside JIT-compiled functions. The `Accumulator` pattern is used to aggregate stats within the JAX loop and return them to Python for saving.
