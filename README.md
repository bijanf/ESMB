<p align="center">
  <img src="docs/figures/logo.svg" alt="Chronos-ESM" width="440">
</p>

<p align="center">
  <b>A differentiable, GPU-ready Earth System Model</b> — research preview (v0.1.0)<br>
  <i>Copyright &copy; 2026 Bijan Fallah · Licensed under <a href="LICENSE">Apache&nbsp;2.0</a></i>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-3b82f6.svg"></a>
  <img alt="status: research preview" src="https://img.shields.io/badge/status-research%20preview-f59e0b.svg">
  <img alt="resolution: T31" src="https://img.shields.io/badge/resolution-T31%20(~3.75%C2%B0)-06b6d4.svg">
  <img alt="built with JAX" src="https://img.shields.io/badge/built%20with-JAX-22d3ee.svg">
</p>

> ⚠️ **Research preview.** This is an early, actively-developed model. Several
> components are validated against observations while others are explicitly
> experimental or carry known biases (see **Project Status** and **Validation**
> below). Read the documented limitations before using results scientifically.

## Overview
Chronos-ESM is a fully differentiable, coupled Earth System Model (Ocean, Atmosphere, Land, Ice) implemented in **JAX**. It is designed for high-performance climate simulation and data assimilation on consumer hardware (GPU/TPU).

### Key Features
*   **Physics**: Primitive-equation ocean (Veros-like, z-levels), spectral atmosphere, thermodynamic sea ice, bucket land.
*   **Technology**: JAX-based end-to-end differentiability for gradient-based tuning.
*   **Resolution**: T31 (~3.75°). Ocean: 15 stretched z-levels to a maximum depth of **5000 m** (layer thickness ~50 m at the surface growing to ~550 m at the bottom). **Variable-depth bathymetry** is derived from ETOPO (full-cell "staircase"); tracer advection/diffusion are flux-masked (no-flux at coasts and the sea floor). The barotropic streamfunction still uses a flat reference depth — topographic steering of the depth-integrated flow is a planned refinement.

![Ocean bathymetry](docs/figures/bathymetry.png)
*   **Performance**: ~280 simulated years per day on a single GPU (T31).

## Project Status (Jun 2026)
Focus has shifted from "does it run" to **"is it right"** — quantitative validation against observations.

*   **Numerics**: Stable (no NaN over multi-thousand-step probes); energy drift bounded and recovering.
*   **Ocean**: Initialised from the WOA18 climatology; preserves realistic large-scale T/S structure. The global **salt budget is closed** (water conservation + global-mean salinity conservation), so salinity no longer drifts into the clip.
*   **Atmosphere**: A correctness overhaul of the single-level spectral dynamics. (1) A **polar instability** — the lat-lon `dx` collapsed at the poles and detonated the surface-pressure field (it pinned to a 99–1206 hPa clamp, variance ~215× observed) — was fixed by flooring the dynamical metric, so **sea-level pressure is now physical** (variance 15× → 2× observed). (2) The winds, which had decayed to a near-standstill (and *anti*-correlated with ERA5), are now realistic — easterly trades, mid-latitude westerlies — via a momentum relaxation that parameterizes the eddy-momentum flux the single level cannot resolve (**surface-zonal-wind correlation −0.08 → 0.72**). (3) Surface geopotential is now persisted in checkpoints, so restarts no longer silently lose topography.
*   **AMOC**: weak and noisy (26.5 °N ≈ 4.6 Sv vs RAPID's ~17; the streamfunction oscillates between snapshots). A multi-agent diagnosis (2026-06-14) showed this is **not** spin-up-limited (the trend is negative). The root blocker is that the ocean velocity field carries a large **spurious net meridional transport** (≈ ±355 Sv globally, where a closed globe requires ~0), so a clean overturning streamfunction is impossible regardless of basin masking — a core ocean-dynamics (barotropic/continuity) issue. Secondary: a fresh subpolar surface lid (no deep convection → no NADW formation) and the absence of vertical tracer advection by *w*. Realistic AMOC is a scoped ocean-model-development task (fix mass conservation → closed-basin/time-mean diagnostic → subpolar restoring for NADW → *w*-advection), not a tuning tweak.
*   **Known biases / limitations** (quantified by the validation framework): the atmosphere is **single-level (barotropic)**, so it has no baroclinic eddies — synoptic systems are absent (sea-level-pressure and meridional-wind *pattern* correlations stay near zero) and the **tropical ITCZ is weak**, leaving precipitation too dry (−1.8 mm/day) and only weakly correlated. A modest warm/over-variable near-surface air-temperature bias also remains. These are the next calibration targets; closing them likely needs vertical structure (a multi-level or external dycore).

## Multi-level atmosphere (experimental — `dinosaur` dycore)
The single-level limitations above are being addressed with a **multi-level** spectral primitive-equation atmosphere built on Google's differentiable `dinosaur` dycore (the core behind NeuralGCM; `chronos_esm/atmos/dino_atmos.py`). Unlike the single level, it performs genuine **baroclinic instability** — growing transient synoptic eddies and a twin upper-tropospheric jet — and carries prognostic moisture for a real ITCZ.

![multi-level dinosaur atmosphere](docs/figures/dino_circulation.png)

*SST-coupled run on the WOA SST (time-mean of a 120-day spin-up): a ~27 m/s eddy-driven baroclinic jet (left), surface winds with synoptic eddy structure (centre), and an equatorial ITCZ (right). Regenerate with `python experiments/dino_circulation_figure.py`.*

Benchmark vs ERA5/WOA18 (SST-forced): it **wins the thermodynamics/hydrology** — precipitation pattern correlation 0.16 → **0.34** with a real ITCZ, plus a better near-surface temperature — but the surface **wind/pressure pattern** skill is not yet improved. At T31 the regridded WOA SST has only a ~22.6 K equator–pole gradient (the sharp Gulf Stream / Kuroshio / ACC SST fronts are smoothed away), too weak for the resolved eddy momentum flux to build the observed mid-latitude surface westerlies (a 22.6 K aquaplanet stays surface-easterly at midlat too). Recovering those needs higher resolution or an explicit eddy-momentum parameterization. The single-level model still provides the **default** coupled atmosphere.

### Running a coupled (dinosaur ↔ ocean) control run
`experiments/run_dino_coupled.py` is a **checkpointing, resumable control-run harness**: it
steps the multi-level atmosphere against the ocean, checkpoints both the ocean state and the
dinosaur modal state every `--ckpt-every-days` (yearly by default), resumes cleanly from any
checkpoint, and writes a **time-mean** of the atmosphere's surface fields into the saved state
so the validation dashboard scores the *dinosaur* atmosphere.
```bash
# fresh 100-year control run (checkpoints to outputs/dino_control/state_d*.nc):
python experiments/run_dino_coupled.py --years 100
# resume toward year 200 from day 36500:
python experiments/run_dino_coupled.py --years 200 --resume 36500
# on the GPU cluster (auto-resumes across back-to-back 23 h jobs):
sbatch experiments/run_dino_control_slurm.sh --years 100
# score the run against WOA18 + ERA5 / refresh the dashboard:
python experiments/make_readme_figures.py "outputs/dino_control/state_d*.nc" --label "dino control"
```
The coupled run holds SST/SSS drift-free via an ocean-only heat & P−E flux balance. Restarts
restore the prognostic state (ocean T/S, modal dynamics) exactly; diagnostic fields (ocean
streamfunction warm-start, density, `w`, and DIC) are recomputed/reset, so a restart is
state-complete but not bit-reproducible (≈0.03 K/week divergence — far below any climate signal).

## Validation
A built-in framework (`chronos_esm/validation/`) scores model output against **WOA18** (ocean T/S) and **ERA5** (near-surface atmosphere), producing area-weighted skill metrics (bias, RMSE, pattern correlation, Taylor statistics), bias maps, and zonal-mean comparisons.

Score any run (or a short in-process demo) against observations:
```bash
# score saved checkpoints (prints/writes the scorecard + figures):
python experiments/validate_control.py --states "outputs/century_physics/year_*.nc" --era5
# or exercise the pipeline on a short in-process run:
python experiments/validate_control.py --demo 200 --era5
# refresh the README validation dashboard below from a saved state (or a glob of
# checkpoints, which is averaged into a climatology):
python experiments/make_readme_figures.py "outputs/century_physics/year_*.nc" --label "years 40-50"
```
ERA5 requires a Copernicus CDS account (`~/.cdsapirc`); WOA18 is fetched/cached automatically.

## Validation dashboard
The maps below are regenerated from the latest simulation state by
`experiments/make_readme_figures.py` (re-run it after a run, commit, and review here).
<!-- VALIDATION:START -->
_Validation of `outputs/atmos_spinup/snap_*.nc` (30-day WOA spin-up (mean of days 20-30; atmosphere overhaul)) against WOA18 + ERA5, auto-generated by `experiments/make_readme_figures.py`. A perfect model has bias 0, corr 1, std ratio 1._

| field | units | bias | RMSE | corr | std ratio | n |
|---|---|---:|---:|---:|---:|---:|
| sst | degC | 0.43 | 2.25 | 0.97 | 1.00 | 2686 |
| sss | psu | 0.13 | 0.87 | 0.84 | 0.67 | 2686 |
| t2m | K | 2.42 | 15.94 | 0.68 | 1.53 | 4608 |
| u_sfc | m/s | 0.61 | 3.17 | 0.72 | 1.26 | 4608 |
| v_sfc | m/s | -0.18 | 1.92 | 0.01 | 0.47 | 4608 |
| precip | mm/day | -1.82 | 3.14 | 0.16 | 0.82 | 4608 |
| mslp | hPa | 2.86 | 18.29 | 0.07 | 2.07 | 4608 |

### Ocean (vs WOA18)
**Sea-surface temperature**
![sst bias map](docs/figures/biasmap_sst.png)
![sst zonal mean](docs/figures/zonal_sst.png)

**Sea-surface salinity**
![sss bias map](docs/figures/biasmap_sss.png)
![sss zonal mean](docs/figures/zonal_sss.png)

### Atmosphere (vs ERA5)
**2 m air temperature**
![t2m bias map](docs/figures/biasmap_t2m.png)
![t2m zonal mean](docs/figures/zonal_t2m.png)

**Surface zonal wind**
![u_sfc bias map](docs/figures/biasmap_u_sfc.png)
![u_sfc zonal mean](docs/figures/zonal_u_sfc.png)

**Surface meridional wind**
![v_sfc bias map](docs/figures/biasmap_v_sfc.png)
![v_sfc zonal mean](docs/figures/zonal_v_sfc.png)

**Precipitation**
![precip bias map](docs/figures/biasmap_precip.png)
![precip zonal mean](docs/figures/zonal_precip.png)

**Mean sea-level pressure**
![mslp bias map](docs/figures/biasmap_mslp.png)
![mslp zonal mean](docs/figures/zonal_mslp.png)

### AMOC (Atlantic overturning)
Model max 11.7 Sv vs RAPID ~17 Sv at 26.5N. (A multi-decade spin-up is needed for a realistic AMOC.)
![AMOC streamfunction](docs/figures/amoc_streamfunction.png)
_(Time series appears once a multi-year run writes yearly checkpoints.)_

<!-- VALIDATION:END -->

## Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/bijanf/ESMB.git && cd ESMB
    ```
2.  Install dependencies:
    ```bash
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    ```

## Usage
Run a WOA-seeded control / spin-up simulation (locally or via SLURM):
```bash
# local:
python experiments/run_century_physics.py --years 100 --ocean-ic woa
# cluster:
sbatch experiments/run_century_physics_slurm.sh --years 100 --ocean-ic woa
tail -f logs/century_v2_*.log
```
Checkpoints are written yearly to `outputs/century_physics/`; resume with `--resume year_NNN`.

## Data
The model's input datasets are **not** committed to the repo — they download
automatically on first use (via `pooch`) and cache to **`~/.cache/chronos_esm/`**:

| Dataset | Used for | Size | Source |
|---|---|---:|---|
| WOA18 T+S (5°) | ocean initial conditions + validation | ~8 MB | NOAA NCEI |
| ETOPO1 bathymetry | ocean land mask + atmosphere orography | ~900 MB | SOCIB THREDDS |
| ERA5 monthly means | *validation only* (`--era5`) | ~MBs | Copernicus CDS (needs `~/.cdsapirc`) |

## Running on a cluster (HPC / SLURM)
Compute nodes usually have **no internet**, so stage the data **once on a login
node** (which does), then submit — the job reads from the shared `~/.cache`:
```bash
git clone https://github.com/bijanf/ESMB.git && cd ESMB     # add auth if the repo is private
module load anaconda cuda/12.3.1                              # site-specific module names
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# GPU JAX (REQUIRED on the cluster): requirements.txt installs CPU JAX, but the
# SLURM script runs on GPU and aborts if no GPU is visible. Install the CUDA build
# matching the installed jax version (uses the bundled nvidia-* wheels):
pip install "jax[cuda12]==$(python -c 'import jax; print(jax.__version__)')"

# 1) stage datasets on the LOGIN node (downloads WOA18 + ETOPO1 to ~/.cache/chronos_esm):
python experiments/prefetch_data.py            # add --era5 to also stage validation data
#    optional full preflight (build the model from cached data before queuing):
python experiments/prefetch_data.py --build

# 2) submit the dinosaur control run (GPU; auto-resumes across back-to-back 23 h jobs):
sbatch experiments/run_dino_control_slurm.sh --years 100
```
The SLURM script runs a JAX-GPU preflight and a `prefetch_data.py --check` cache
guard, so a job started without staged data fails fast with instructions rather
than hanging on a blocked download. **Adjust for your site:** the `#SBATCH`
account/partition/qos lines (defaults `--account=poem --partition=gpu
--qos=gpumedium`) and the `module load` names. If your `$HOME` quota is small, set
`export XDG_CACHE_HOME=/path/on/shared/scratch` **before both** `prefetch_data.py`
and `sbatch` (the cache must be the same path, visible from login and compute nodes).

## Citing
If you use Chronos-ESM in academic work, please cite it via [`CITATION.cff`](CITATION.cff)
(GitHub renders a "Cite this repository" button from it).

## Contributing
Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for the dev setup,
tests, and linters, and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for community norms.

## License
Licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
You may use, modify, and redistribute it under those terms; it is provided "as is",
without warranty.
