# Prognostic-momentum ocean core (P3)

**Goal:** replace the ocean's *diagnostic* velocities (wind-driven flat-bottom Stommel
barotropic + thermal-wind baroclinic, with the depth-mean removed) and the interim
Stommel/box thermohaline closure with a genuinely **time-integrated overturning**, so that

1. density has a real dynamical pathway to the overturning (`d(AMOC)/d(density) ≠ 0` for the
   *right* reason, not via a parameterized closure), and
2. the AMOC bifurcation ([amoc_tipping.md](amoc_tipping.md)) becomes a **clean** dynamical
   saddle-node instead of the present ±10 Sv relaxation-oscillation of the box closure.

This is the critical path to a publishable working forcing-response ESM.

## Why the current ocean can't do it

`chronos_esm/ocean/veros_driver.py:step_ocean` sets `u, v` diagnostically (thermal
wind/geostrophy) and removes the per-latitude net transport; there is **no `du/dt`**. With
a flat-bottom streamfunction over variable bathymetry the depth-integrated flow is not
divergence-free, which is patched by a per-latitude corrector. The verified baseline (P0)
is `AMOC@26.5°N ≈ 0` and `d(AMOC)/d(subpolar freshening) = 0` — density cannot drive an
overturning, so no forcing can move one. The thermohaline closure (P4) is an *interim*
work-around, not a fix.

## Architecture (phased)

| Stage | What | Status |
|---|---|---|
| **S2** | AD-safe variable-coefficient elliptic invert `div(coef·∇ψ)=rhs` (Cartesian + spherical), `coef=1/H` → JEBAR/topographic operator | **done** |
| **S3** | Prognostic barotropic vorticity (`∂ζ/∂t = −β·v − r·ζ + curl(τ)/(ρH)`), ψ inverted each step; validated against the Stommel/Sverdrup gyre | **done (standalone)** |
| **S4** | Prognostic baroclinic `du/dt` (semi-implicit Coriolis, hydrostatic pressure gradient, viscosity, friction) — the AMOC-relevant density-driven overturning | **done (standalone)** |
| **S5** | Couple S3 (barotropic) + S4 (baroclinic) and wire into `step_ocean` **behind a flag**; retire the THC closure once the dynamical chain carries overturning to 26.5°N; re-run the hysteresis sweep for a clean bifurcation | **first increment wired (flagged)** |

### S5 status (first increment)

`step_ocean(..., prognostic_momentum=True)` (a static jit argument, **default off → zero
regression**) advances the baroclinic velocity prognostically via `momentum.step_momentum`
instead of the diagnostic thermal wind; `state.u/v` carry it across steps. Only the
baroclinic part is kept (wet-column mean removed) so the unbalanced barotropic mode cannot
run away; the wind-driven barotropic streamfunction and the net-transport corrector are
retained for now.

**Verified** (`tests/test_prognostic_momentum.py`): this breaks the P0 blocker —
`d(overturning)/d(subpolar salt) ≈ +56` with the prognostic momentum vs `≈ +0.12` for the
diagnostic thermal wind (smooth RMS-overturning metric), i.e. density now drives the
overturning ~470× more strongly; a 15-step run stays finite.

**Remaining S5 work (not yet done):** the prognostic flow is over-energetic after a short
spin-up (needs drag tuning + a long equilibration before it is a realistic production AMOC);
the proper barotropic-streamfunction solve (replacing the crude per-latitude corrector with
the rigid-lid elliptic projection from S2) and retiring the interim THC closure are the next
increments, followed by re-running the hysteresis sweep for a clean (low-noise) bifurcation.

## What's built (S2–S4)

All in pure JAX, differentiable, **standalone — not yet wired into the live model** (zero
regression). Validated by `tests/test_barotropic_gyre.py` (9 tests) and
`tests/test_momentum.py` (3 tests).

`chronos_esm/ocean/solver.py`
- `solve_elliptic_varcoef(coef, rhs, dx, dy, mask, x0)` — Cartesian `div(coef·∇ψ)=rhs`.
- `apply_elliptic_varcoef(...)` — applies the operator (for manufactured solutions / a
  viscous `∇²ζ` term).
- `solve_elliptic_varcoef_sphere(coef, rhs, lat, dlon, dlat, a, mask, x0)` — the lat-lon
  operator `(1/(a²cosφ))[∂_λ(coef/cosφ ∂_λψ) + ∂_φ(coef cosφ ∂_φψ)] = rhs`.
  - **Symmetric face-conductance discretization** (multiply the cell equation by the cell
    area `a²cosφ dλ dφ`) so the matrix stays SPD and the existing scan-CG applies; the
    area-divided form used for the Cartesian operator is non-symmetric once the cell area
    varies with latitude.
  - Dirichlet `ψ=0` at coasts via the land-identity rows — do **not** mask the faces (that
    imposes a singular Neumann condition and CG diverges; caught by a random-RHS test).
  - Polar `cosφ` floor avoids the `1/cosφ` blow-up at the lat-lon pole singularity.

`chronos_esm/ocean/barotropic.py`
- `step_barotropic` / `spin_up_gyre` (Cartesian) and `step_barotropic_sphere` /
  `spin_up_gyre_sphere` (lat-lon): prognose ζ, diagnose ψ via the elliptic invert. On the
  sphere the planetary-vorticity advection simplifies to `β·v = (2Ω/a²)∂_λψ` (cosφ cancels).
- `wind_stress_curl[_sphere]`, `velocities_from_psi` / `velocities_sphere`.

`chronos_esm/ocean/momentum.py` (S4)
- `step_momentum` / `spin_up` — prognostic baroclinic `du/dt`, `dv/dt` with
  **semi-implicit Coriolis** (`coriolis_semi_implicit`, a 2×2 rotation solve — stable for
  any `dt·f`; explicit `f×u` is unconditionally unstable), the **hydrostatic
  pressure-gradient force** from density (`hydrostatic_pressure` → `pressure_gradient_accel`),
  linear drag, and optional horizontal viscosity. Validated against **thermal-wind balance**
  (`du/dz = −(g/(ρ₀f)) ∂ρ/∂y`) and Coriolis stability at `dt·f = 5`; differentiable.

`chronos_esm/ocean/diagnostics.py`
- `compute_barotropic_streamfunction` — the x-y transport streamfunction (gyre validation),
  distinct from the depth-latitude AMOC/MOC.

## Known issues / what remains

- **Preconditioner.** The anisotropic spherical operator (`cE ∝ 1/cosφ`, `cN ∝ cosφ`) is
  poorly conditioned for Jacobi-CG — the residual stalls at a float64 floor while the
  *solution* is accurate (~1e-11). Fine for unit tests via warm-start; a stronger
  preconditioner (line-relaxation / multigrid) is needed before long production runs.
- **S4 stability.** Coriolis must be semi-implicit (explicit `f×u` is unconditionally
  unstable); the equator needs friction/viscosity once the `|f|` clamp is removed; the
  A-grid layout may permit checkerboard noise (watch the gyre test; Ah/shapiro first, C-grid
  migration only if forced).
- **AMOC sensitivity during transition.** Keep the THC closure ON through S3/S4 and retire it
  only in S5 after the dynamical `d(AMOC)/d(density)` is verified nonzero — otherwise there is
  a window with no overturning sensitivity at all.

## Validation

```bash
PYTHONPATH=$PWD JAX_PLATFORMS=cpu venv/bin/python -m pytest tests/test_barotropic_gyre.py -q
```
Asserts: manufactured-solution round trips (constant + variable coef, Cartesian + spherical,
to ~1e-11), agreement with `solve_poisson_2d`, Cartesian and spherical **Stommel gyres**
(interior Sverdrup balance `β·v = curl(τ)/(ρH)` + western intensification), and `jax.grad`
of the gyre strength is finite and nonzero.
