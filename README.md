<p align="center">
  <img src="docs/figures/logo.svg" alt="Chronos-ESM" width="440">
</p>

<p align="center">
  <b>A differentiable, GPU-ready Earth System Model</b> — research preview (v0.1.0)<br>
  <i>Copyright &copy; 2026 Bijan Fallah · Licensed under <a href="LICENSE">Apache&nbsp;2.0</a></i>
</p>

<p align="center">
  <a href="https://github.com/bijanf/ESMB/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/bijanf/ESMB/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/bijanf/ESMB/releases/latest"><img alt="latest release" src="https://img.shields.io/github/v/release/bijanf/ESMB?color=10b981"></a>
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache--2.0-3b82f6.svg"></a>
  <img alt="status: research preview" src="https://img.shields.io/badge/status-research%20preview-f59e0b.svg">
  <img alt="resolution: T31" src="https://img.shields.io/badge/resolution-T31%20(~3.75%C2%B0)-06b6d4.svg">
  <img alt="built with JAX" src="https://img.shields.io/badge/built%20with-JAX-22d3ee.svg">
  <!-- After archiving the repo on Zenodo (zenodo.org → GitHub settings → flip ESMB on,
       then re-publish the release), uncomment the concept-DOI badge Zenodo gives you:
  <a href="https://doi.org/10.5281/zenodo.XXXXXXX"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg"></a>
  -->
</p>

> ⚠️ **Research preview.** This is an early, actively-developed model. Several
> components are validated against observations while others are explicitly
> experimental or carry known biases (see **Project Status** and **Validation**
> below). Read the documented limitations before using results scientifically.

> 🚀 **What's new in [v0.1.0](https://github.com/bijanf/ESMB/releases/tag/v0.1.0)** — the
> first public research preview. Four working-ESM milestones, each validated and
> documented honestly: a multi-level **`dinosaur`** atmosphere, a **CO₂ forcing
> response** (+1.58 K), a verified **AMOC tipping point** (~[0.38, 0.75] Sv hosing),
> and a **mid-Holocene (6 ka)** paleo run with a real seasonal cycle. See the
> [release notes](docs/release_notes_v0.1.0.md) and the **[showcase gallery](docs/showcase.md)**.

### Showcase
Baroclinic eddies emerging from rest in the multi-level `dinosaur` dycore (Held–Suarez
aquaplanet, lower-tropospheric relative vorticity) — genuine baroclinic instability,
the dynamics the single-level atmosphere cannot produce:

![baroclinic eddies from rest](docs/figures/dino_baroclinic_vorticity.gif)

More results — the AMOC tipping hysteresis, the CO₂ response, the 6 ka orbital
fingerprint, and the full validation dashboard — are in the **[showcase gallery](docs/showcase.md)**.

## Overview
Chronos-ESM is a fully differentiable, coupled Earth System Model (Ocean, Atmosphere, Land, Ice) implemented in **JAX**. It is designed for high-performance climate simulation and data assimilation on consumer hardware (GPU/TPU).

> 📘 **Manual / textbook.** A book-length, equation-complete manual — every governing equation derived from the code, the differentiable-modeling paradigm, canonical climate test cases (AMOC tipping, jets, ITCZ), a CMIP7 roadmap, and a teaching syllabus — is in [`docs/manual/`](docs/manual/). Read the compiled PDF: [**`docs/manual/main.pdf`**](docs/manual/main.pdf) (rebuild with `docs/manual/build.sh`).

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
*   **AMOC**: under active development (ocean core, "P3"). Several fixes have landed: the spurious net meridional transport was corrected (≈ ±330 → 0.000 Sv via a per-latitude barotropic corrector); the overturning-streamfunction **sign bug** was fixed (the diagnostic was scoring a physically-correct ~15 Sv cell as ~0 Sv — verified by injecting a known cell, so the earlier "≈0 / incoherent" readings were largely a metric artifact); and vertical tracer advection by *w* was added. An interim density-driven **thermohaline closure** now gives an AMOC ≈ 15 Sv at 26.5 °N that **responds to density** — `d(AMOC)/d(subpolar salinity)` is nonzero, sign-correct (saltier/denser subpolar → stronger AMOC), and differentiable. A tunable salt-advection feedback on this closure produces a **verified AMOC tipping point**: a genuine saddle-node hysteresis window ≈ **[0.38, 0.75] Sv** of subpolar freshwater hosing (centered ~0.6 Sv), confirmed by an initial-condition (on-state vs off-state) bistability test — the branches stay 8–9 Sv apart after 100 yr, statistically significant (see [AMOC tipping](docs/amoc_tipping.md), `docs/figures/amoc_bistability.pdf`). The branches are still noisy (±~10 Sv relaxation-oscillation). A fully **prognostic-momentum ocean core** (baroclinic `du/dt` + rigid-lid projection) was built, wired behind a flag, and proven density-responsive, but calibration revealed a **T31 resolution barrier**: at ~3.75° the geostrophic overturning is O(300–550 Sv) — ~20–40× too large — not tunable by drag, and fixable only by mesoscale-eddy parameterization + finer resolution (the full ocean-dynamical-core project). The production AMOC therefore ships with the working diagnostic + thermohaline closure (which *is* density-driven and delivered the tipping result above); the prognostic core stays research-in-progress (see [prognostic ocean core](docs/prognostic_ocean_core.md)). The validation dashboard below scores the **130-yr dinosaur coupled control** (refreshed via `experiments/score_dino_control.py`).
*   **Forcing response (CO2)**: in free-ocean (frozen q-flux) mode the coupled model **warms under CO2** — an abrupt-2×CO2 experiment gives ΔSST ≈ **+1.58 K** (0.43 K/(W/m²)). This is a transient, surface-forcing **proxy**, not equilibrium climate sensitivity (no atmospheric radiative-feedback amplification; cold-tropics base state).
*   **Paleo (mid-Holocene, 6 ka)**: the model has a **real seasonal cycle** (orbital insolation, `chronos_esm/orbital.py`) and responds to **orbital boundary conditions** with the correct mid-Holocene fingerprint — a PMIP-style 6 ka-vs-PI run (orbit-only difference) gives **NH summer warming +1.1 K (20–60 °N), +1.9 K Arctic** and **monsoon intensification** where one is resolved (S/SE Asia +31 %, N. America +20 % JJA precip; ITCZ +0.2° north), with global-annual ≈ 0 as orbital forcing requires. Differentiable (`d(insolation)/d(obliquity)`). The "Green Sahara" is not captured (the T31 African monsoon is ~absent to begin with). See [mid-Holocene experiment](docs/paleo_midholocene.md), `docs/figures/paleo_midholocene.pdf`.
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

## Validation dashboards

Two complementary, auto-scored assessments against **WOA18** (ocean T/S) and **ERA5**
(near-surface atmosphere) — a perfect model has bias 0, corr 1, std ratio 1:

1. **Coupled control** — a 100-year coupled `dinosaur`↔ocean run, **flux-corrected**
   (surface SST restored toward WOA), scored over the equilibrated years 82–100.
2. **Atmosphere-only (forced)** — isolates the atmosphere physics: a 30-day spin-up
   with SST held to WOA18, so coupled-ocean bias can't mask atmospheric skill.

Regenerate either from saved checkpoints with `experiments/make_readme_figures.py`.

### Coupled control — 100-yr flux-corrected (dinosaur ↔ ocean)
A fully coupled century (multi-level `dinosaur` spectral atmosphere ↔ z-level ocean),
**flux-corrected**: the surface heat flux restores SST toward WOA (Haney, τ=30 d) to
remove a structural cold bias, while the atmosphere, land surface and precipitation
evolve freely. This run also closes the sub-grid Central-American isthmus (Panama) and
adds an interactive slab land surface. Stable over 100 years (SST ≈18.1 °C and SSS ≈34.7
flat; baroclinic eddies under the wind clamp), scored over the equilibrated years 82–100.
<!-- CONTROL:START -->
_Flux-corrected 100-yr coupled control (`dinosaur` ↔ ocean; SST restored to WOA), years 82–100, vs WOA18 + ERA5._

| field | units | bias | RMSE | corr | std ratio | n |
|---|---|---:|---:|---:|---:|---:|
| sst | degC | 0.44 | 0.89 | 1.00 | 1.00 | 2686 |
| sss | psu | 0.07 | 1.57 | 0.10 | 0.41 | 2686 |
| t2m | K | 3.52 | 8.63 | 0.88 | 0.57 | 4608 |
| u_sfc | m/s | -0.99 | 4.05 | 0.10 | 0.59 | 4608 |
| v_sfc | m/s | -0.12 | 2.24 | 0.15 | 0.98 | 4608 |
| precip | mm/day | -2.32 | 3.07 | 0.37 | 0.45 | 4608 |
| mslp | hPa | -11.70 | 15.87 | -0.35 | 0.60 | 4608 |

Honest read: the headline gains over the earlier free-running control are a **revived
ITCZ** (precip pattern corr **0.08 → 0.37**, now raining on the equator) and **land air
temperature** (t2m corr **0.50 → 0.88**, once continents stopped acting as a 1 °C cold
ocean). Note SST corr is **1.00 by construction** — it is *restored* toward WOA — so SST
and the SST-driven part of t2m are imposed, not predicted; the genuinely emergent skill is
in precipitation and the land/ocean contrast. Mean sea-level pressure stays the weakest
field (corr −0.35): the T31 single-gradient atmosphere has little synoptic skill.

#### Ocean (vs WOA18)
**Sea-surface temperature**
![sst bias map](docs/figures/control/biasmap_sst.png)
![sst zonal mean](docs/figures/control/zonal_sst.png)

**Sea-surface salinity**
![sss bias map](docs/figures/control/biasmap_sss.png)
![sss zonal mean](docs/figures/control/zonal_sss.png)

#### Atmosphere (vs ERA5)
**2 m air temperature**
![t2m bias map](docs/figures/control/biasmap_t2m.png)
![t2m zonal mean](docs/figures/control/zonal_t2m.png)

**Surface zonal wind**
![u_sfc bias map](docs/figures/control/biasmap_u_sfc.png)
![u_sfc zonal mean](docs/figures/control/zonal_u_sfc.png)

**Surface meridional wind**
![v_sfc bias map](docs/figures/control/biasmap_v_sfc.png)
![v_sfc zonal mean](docs/figures/control/zonal_v_sfc.png)

**Precipitation**
![precip bias map](docs/figures/control/biasmap_precip.png)
![precip zonal mean](docs/figures/control/zonal_precip.png)

**Mean sea-level pressure**
![mslp bias map](docs/figures/control/biasmap_mslp.png)
![mslp zonal mean](docs/figures/control/zonal_mslp.png)

#### AMOC (Atlantic overturning)
Model max 34.2 Sv vs RAPID ~17 Sv at 26.5°N. This "max" is a **noisy single-snapshot**
maximum: the 26.5°N cell is a clean textbook shape at any instant (smooth ~20 Sv max near
900 m, closing at the floor), but its *amplitude* oscillates strongly over the run while SST
stays flat (so it is **not** an SST-drift signal). The diagnostic thermal wind is negligible
(~0.1 Sv, drag-damped); the AMOC is the density-driven **thermohaline closure**, which sets
the overturning *instantaneously* from the small subpolar−subtropical density contrast and so
has **no temporal inertia**. (The earlier "spurious net transport" is fixed — basin net
transport is ~0.) Time-mean AMOC is ≈20 Sv; the planned fix adds multi-year inertia to the
closure (relax the overturning amplitude over τ~1–3 yr) — see
[`experiments/diagnose_coupled_amoc.py`](experiments/diagnose_coupled_amoc.py).
![AMOC streamfunction](docs/figures/control/amoc_streamfunction.png)
<!-- CONTROL:END -->

### Atmosphere-only — forced 30-day WOA spin-up
With SST held to WOA18 the atmosphere physics is graded in isolation (no coupled-ocean
bias to hide behind). The block below is auto-generated by
`experiments/make_readme_figures.py` (re-run it after a forced spin-up, commit, and review here).
<!-- VALIDATION:START -->
_Validation of the dino control run (130 yr) against WOA18 + ERA5, auto-generated by `experiments/make_readme_figures.py`. A perfect model has bias 0, corr 1, std ratio 1._

| field | units | bias | RMSE | corr | std ratio | n |
|---|---|---:|---:|---:|---:|---:|
| sst | degC | 0.63 | 1.18 | 1.00 | 0.96 | 2686 |
| sss | psu | 0.04 | 1.58 | 0.01 | 0.31 | 2686 |
| t2m | K | 3.72 | 8.81 | 0.88 | 0.56 | 4608 |
| u_sfc | m/s | -0.95 | 4.07 | 0.09 | 0.60 | 4608 |
| v_sfc | m/s | -0.10 | 2.36 | 0.15 | 1.08 | 4608 |
| precip | mm/day | -2.34 | 3.12 | 0.33 | 0.48 | 4608 |
| mslp | hPa | -11.71 | 15.86 | -0.35 | 0.59 | 4608 |

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
Model max 75.8 Sv vs RAPID ~17 Sv at 26.5N. This "max" is a **noisy single-snapshot** maximum of the overturning streamfunction; the cell is a clean ~20 Sv shape at any instant but its amplitude oscillates (the density-driven thermohaline closure responds instantaneously, with no temporal inertia — basin net transport is ~0, already fixed). Time-mean AMOC is ≈20 Sv; the planned fix adds multi-year inertia to the closure (see `experiments/diagnose_coupled_amoc.py`).
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
# checkpoints are written to SCRATCH, not home (PIK default /p/tmp/$USER/dino_control;
# override with CHRONOS_OUTDIR=/your/scratch/path). Score them with:
python experiments/make_readme_figures.py "/p/tmp/$USER/dino_control/state_d*.nc" --label "dino control"
```
The SLURM script runs a JAX-GPU preflight and a `prefetch_data.py --check` cache
guard, so a job started without staged data fails fast with instructions rather
than hanging on a blocked download. It writes model output to **scratch**
(`$CHRONOS_OUTDIR`, default `/p/tmp/$USER/dino_control`) — never home. **Adjust for your site:** the `#SBATCH`
account/partition/qos lines (defaults `--account=poem --partition=gpu
--qos=gpumedium`) and the `module load` names. If your `$HOME` quota is small, set
`export XDG_CACHE_HOME=/path/on/shared/scratch` **before both** `prefetch_data.py`
and `sbatch` (the cache must be the same path, visible from login and compute nodes).

## Citing
If you use Chronos-ESM in academic work, please cite it via [`CITATION.cff`](CITATION.cff)
(GitHub renders a "Cite this repository" button from it).

**Please also cite the component sources.** Chronos-ESM is a differentiable
coupling and diagnostics layer wrapping mostly standard, published physics and
third-party data. [`docs/ATTRIBUTION.md`](docs/ATTRIBUTION.md) maps every
component to its origin (genuine vs. borrowed) and citation, with full BibTeX in
[`docs/manual/references.bib`](docs/manual/references.bib). In particular, any
result using the bundled datasets must cite **WOA18** (Locarnini et al. 2018;
Zweng et al. 2018), **ERA5** (Hersbach et al. 2020), and **ETOPO1** (Amante &
Eakins 2009), and the multi-level atmosphere relies on the **`dinosaur`** dycore
(Kochkov et al. 2024).

**Getting a DOI (recommended for citability):** archive the repository on
[Zenodo](https://zenodo.org) — sign in with GitHub, flip **ESMB** on under
*Settings → GitHub*, then (re-)publish a GitHub release. Zenodo mints a permanent
concept DOI; add it to the badge row above and to the `identifiers:` field in
[`CITATION.cff`](CITATION.cff). Release metadata for Zenodo is in
[`.zenodo.json`](.zenodo.json).

## Contributing
Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for the dev setup,
tests, and linters, and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for community norms.

## License
Licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
You may use, modify, and redistribute it under those terms; it is provided "as is",
without warranty.
