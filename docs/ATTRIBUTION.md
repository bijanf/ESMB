# Attribution & Provenance

Chronos-ESM is a *differentiable coupling and diagnostics layer* wrapping mostly
**standard, published component physics**. This document credits every method,
parameterization, dataset, and software dependency the model re-implements or
ingests, and states honestly what is genuinely original versus borrowed. Full
BibTeX entries are in [`docs/manual/references.bib`](manual/references.bib).

If you use Chronos-ESM, please cite the model (see [`CITATION.cff`](../CITATION.cff))
**and** the relevant component sources below — especially the data products.

---

## How much is ours vs. borrowed (the honest summary)

The **bulk of the physics is standard and borrowed**, and that is normal for a
model of this kind. The active atmosphere *is* a third-party dependency (Google's
`dinosaur` dycore). Nearly every parameterization is textbook or published.

What is **genuinely Chronos-ESM's own is systems-level engineering, not new physics**:

1. **End-to-end differentiable JAX coupling** of a z-level ocean + bucket land +
   Semtner ice around the (third-party, already-differentiable) `dinosaur`
   atmosphere, in a single jittable/gradable interval. *Adjacent prior art that
   we build on / sit beside:* NeuralGCM/`dinosaur` (the atmosphere we wrap;
   `kochkov2024neuralgcm`) and **Veros** (a differentiable JAX ocean GCM;
   `hafner2018veros`). The novelty is the differentiable **ocean + coupler + ice +
   land** stack and the coupling — *not* differentiability per se, and *not* the
   atmospheric autodiff.
2. **THC-inertia coupled-AMOC fix** — the *diagnosis* that coupled-AMOC noise comes
   from an inertia-less closure, and the choice to relax a carried overturning
   amplitude toward the density-implied target over a multi-year timescale. The
   relaxation operator itself is a standard first-order filter; the contribution is
   the diagnosis and application.
3. **Per-column AMOC diagnostic + sign fix** — correct barotropic removal per wet
   column over variable bathymetry, and a RAPID-convention sign fix. This is *correct
   handling of a coarse-box artifact*, i.e. an engineering fix, not a field-level
   advance over community MOC practice.
4. **Differentiable paleo / proxy / DA scaffolding** — orbital forcing
   differentiable in obliquity/eccentricity/precession, and proxy forward
   operators + masked losses for gradient-based fitting (currently partial).
5. **AD-safe smooth substitutions** for non-differentiable physics (soft slope
   limiter, smooth convection ramp, softplus precip trigger, sigmoid ice albedo,
   straight-through T/S clips) — a consistent application of standard
   differentiable-programming relaxations throughout the model.

Everything else below is credited to its originators.

---

## Component provenance table

**Legend** — Origin class: `dep` = third-party dependency · `std` = textbook/standard
method (our code) · `pub` = published parameterization (our code) · `data` = ingested
dataset · `orig` = Chronos-ESM original.

### Ocean

| Component | Origin class | Source / citation |
|---|---|---|
| Linear equation of state | std | `vallis2017`, `griffies2004` |
| RK4 tracer stepping; upwind vertical advection | std | classical numerical analysis |
| Biharmonic / Shapiro filter | std | `shapiro1970` |
| Barotropic (wind-gyre) streamfunction; CG Poisson solve | pub | `stommel1961`, `sverdrup1947` |
| Thermal-wind / hydrostatic diagnostic velocities | std | `vallis2017` |
| Gent–McWilliams eddy bolus (`gm.py`) | pub | `gentmcwilliams1990` + slope taper `danabasoglu1995` (DM95) + surface taper `large1997` (LDD97) |
| Isoneutral (Redi) diffusion tensor (`mixing.py`, inactive) | std | `redi1982` |
| Convective adjustment (smooth ramp) | std/adapt | `rahmstorf1993`, `marotzke1991` (smooth ramp is our AD-safe adaptation) |
| THC overturning closure (`overturning.py`, interim) | pub | `stommel1961`, `marotzke1991` |
| **THC-inertia coupled-AMOC fix** | **orig** | Chronos-ESM original |
| **Per-column AMOC metric + sign fix** | **orig** | Chronos-ESM (convention: `griffies2004`) |
| Bathymetry hFac / shaved-cell wet masks | pub | `adcroft1997` |
| Surface restoring / q-flux correction; salinity sponge | pub | `haney1971` |
| Differentiable JAX ocean (adjacent prior art) | — | `hafner2018veros` (Veros) |

### Atmosphere

| Component | Origin class | Source / citation |
|---|---|---|
| Multi-level spectral dycore (active path) | dep | `kochkov2024neuralgcm` + software `dinosaur_software` |
| dino init / vorticity seed / weak-diffusion tuning (`dino_atmos.py`) | orig | Chronos-ESM (the wrapper, not the dycore) |
| Held–Suarez forcing (dycore test) | pub | `heldsuarez1994` |
| Single-level barotropic dynamics (`dynamics.py`, legacy) | std/pub | `jablonowski2006`, `arakawalamb1981`, `sadourny1975` |
| QTCM (`qtcm.py`, dead code) | pub | `neelinzeng2000` |

### Coupler / Land / Ice

| Component | Origin class | Source / citation |
|---|---|---|
| **Differentiable coupling step** (`dino_step.py`) | **orig** | Chronos-ESM original |
| Bulk aerodynamic fluxes (SH/LH/τ, drag coeff.) | pub | `largeyeager2004` |
| Gauss↔linear regridding | std | standard interpolation |
| Bucket land hydrology | pub | `manabe1969` |
| Semtner sea-ice thermodynamics (partial: freezing-floor) | pub | `semtner1976` |

### Forcing / Paleo / Validation / Proxy

| Component | Origin class | Source / citation |
|---|---|---|
| Orbital daily-mean insolation (`orbital.py`) | pub | `berger1978` |
| PMIP4 orbital parameter values | data/pub | `ottobliesner2017` |
| CO₂ radiative forcing | pub | `myhre1998` |
| Historical CO₂ table | data | `noaagml_co2`, `ipcc2021ar6` |
| Total solar irradiance (1361 W/m²) | data | `kopplean2011` |
| Taylor-diagram skill metrics | std | `taylor2001` |
| Proxy δ¹⁸O foram operator (only working operator) | pub | `bemis1998` |
| **Differentiable proxy / 4D-Var scaffolding** (partial) | **orig** | Chronos-ESM original |
| CMIP experiment framing | std | `eyring2016cmip6` |

### Datasets (ingested — cite in any publication)

| Data product | Used for | Citation |
|---|---|---|
| World Ocean Atlas 2018 (T, S) | Ocean ICs, SST/SSS restoring, validation | `locarnini2018woa`, `zweng2018woa` |
| ERA5 reanalysis | Atmospheric validation benchmark | `hersbach2020era5` |
| ETOPO1 relief | Orography, ocean depth, wet masks | `amante2009etopo` |
| RAPID-MOCHA 26.5°N | AMOC benchmark (~17 Sv) | `smeed2018` |

### Software stack

JAX (`jax2018`), `dinosaur` (`dinosaur_software`), NumPy (`harris2020numpy`),
SciPy (`virtanen2020scipy`), xarray (`hoyer2017xarray`), Matplotlib
(`hunter2007matplotlib`), Pooch (`uieda2020pooch`), Optax (`deepmind2020jax`),
plus netCDF4 and cdsapi/Copernicus CDS. See [`NOTICE`](../NOTICE) for licenses.

---

## What we explicitly do **not** claim

- We do **not** claim novel atmospheric dynamics — the atmosphere is `dinosaur`.
- We do **not** claim a new eddy, mixing, ice, land, or convection scheme — all are
  standard/published and cited above.
- We do **not** claim the per-column MOC removal or the THC relaxation as new
  *methods* — they are correct engineering choices using standard operators.
- AMOC tipping in Chronos-ESM emerges from a **prescribed-shape, density-scaled THC
  closure** with a tunable bistability, **not** from resolved prognostic momentum.

Corrections and additional attribution requests are welcome — please open an issue.
