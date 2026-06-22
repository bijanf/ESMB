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
| **S5** | Couple S3 (barotropic) + S4 (baroclinic), wire into `step_ocean` **behind a flag**, rigid-lid project; calibrate to a realistic AMOC, retire the THC closure | **wired (flagged); calibration shows a T31 resolution barrier — see below** |

### S5 status — wired, but blocked by a coarse-resolution physics barrier (2026-06-22)

`step_ocean(..., prognostic_momentum=True)` (a static jit argument, **default off → zero
regression**) advances the baroclinic velocity prognostically via `momentum.step_momentum`
instead of the diagnostic thermal wind; `state.u/v` carry it across steps; the depth-integrated
flow is made non-divergent with the **rigid-lid elliptic projection** (`barotropic.rigid_lid_project`,
solve `∇²χ=∇·(U,V)`, subtract `∇χ/H`) that supersedes the crude per-latitude corrector.

**The density-responsiveness milestone stands** (`tests/test_prognostic_momentum.py`): this
breaks the P0 blocker — `d(overturning)/d(subpolar salt) ≈ +56` with the prognostic momentum
vs `≈ +0.12` for the diagnostic thermal wind (smooth RMS-overturning metric), ~470× stronger;
mass conservation holds (`max|∇·U| < 1e-10`); runs stay finite.

**But the equilibrated coupled AMOC is not calibratable to realism at T31 by drag.** A 12-yr
coupled drag sweep (10 / 3 / 1-day momentum drag, THC off) plus an algebraic analysis of the
same equilibrated density field (`experiments/diagnose_prognostic_amoc.py`) showed:

- The run logs' "AMOC ≈ 0 Sv" was a **metric artifact** (`upper_cell = max(profile)` of an
  all-negative profile). The true 26.5°N cell is a coherent but **reversed, ~10–40× too strong**
  overturning (−222 Sv at 10-day drag, −132 at 3-day) — drag scales its amplitude but not its
  structure/sign.
- The AMOC magnitude is set entirely by the **momentum regime** the drag selects. Sweeping the
  drag `r` on the *same* density field:

  | drag `r` | τ | regime (f²≈1e-8 vs r²) | AMOC upper |
  |---|---|---|---|
  | 5e-2 /s | **20 s** | r²≫f² → drag-damped creep | **+1 Sv** |
  | 1e-3 /s | 1000 s | drag-damped | +50 Sv |
  | 1.16e-6 /s | **10 d** | f²≫r² → **geostrophic** | **+309 Sv** |
  | 3e-5 /s | 0.4 d | geostrophic | +553 Sv |

- **The production diagnostic path is usable only because it sets `r_drag = 0.05/s` (a 20-second
  Rayleigh drag).** That is *not geostrophy* — it damps the thermal-wind flow ~500× (`~f/r`) into
  a smooth down-pressure-gradient creep (~+1 Sv), on top of which the **THC closure adds the
  real ~15 Sv** density-driven Atlantic cell. The prognostic core, run at a *realistic* (10-day)
  drag, is genuinely geostrophic → the T31 density field produces an unusable 300–550 Sv overturning.

**Root cause — a resolution barrier, not a tuning knob.** At T31 (~3.75°) the geostrophic
thermal-wind transports are O(20–40×) too large. A realistic *prognostic* AMOC needs the physics
that real models use to tame this: **Gent–McWilliams mesoscale-eddy parameterization**, a proper
**rigid-lid surface pressure** the baroclinic flow is referenced against, and likely **finer
resolution**. That is the full ocean dynamical-core project — weeks of work — not a drag calibration.

### GM implemented + measured (2026-06-22): necessary, but NOT the dominant lever

`chronos_esm/ocean/gm.py` now provides a correct, AD-safe Gent–McWilliams eddy parameterization
(latitude-aware isopycnal slopes with a sign-preserving ε-floor; soft slope clip; **DM95
steep-slope taper + a surface taper**; bolus velocity `u*=∂_z(κ_eff S)` with `Ψ*=0` at the
surface/floor so it is depth-integral-zero), wired into `step_ocean(gm_on=True)` (default off →
zero regression; supersedes the untapered `mixing.py` bolus). Tests: `tests/test_gm.py` (6).

Measured on the **WOA18 T31 ocean** (`experiments/diagnose_gm_amoc.py`):

| quantity | value |
|---|---|
| WOA isopycnal slope `\|Sy\|` (median / p90) | 1.7e-4 / 2.2e-3 → **GM active** (gentle, not tapered off) |
| GM bolus `\|v*\|` (max / mean) | 1.2e-2 / 3.5e-4 m/s → eddy overturning **~1–2 Sv** (correct Atlantic magnitude) |
| AMOC upper cell 26.5 N, Eulerian-mean (GM off) | **326 Sv** (the barrier) |
| AMOC upper cell 26.5 N, **residual-mean** (Eulerian + GM bolus) | **324 Sv** (reduction **~0.5 %**) |
| `d(AMOC)/d(subpolar salt)`, GM on | +33 Sv/psu (density pathway **preserved**) |

**Finding:** GM is correctly implemented and produces a physically-right ~1–2 Sv eddy (bolus)
overturning, but it reduces the 326 Sv barrier by only ~1 %. So GM is *necessary* eddy physics
but **not the lever for this barrier**; the earlier "GM cuts the overturning toward ~15 Sv"
expectation was optimistic at T31. (The slow isopycnal-*flattening* mechanism operates on
multi-decadal timescales not testable here; even at literature-typical magnitude it would not
close a 20× gap.)

### Surface-pressure reference settled (2026-06-22): NOT the lever either — the barotropic mode is already clean

The other roadmap-candidate lever was a rigid-lid **surface-pressure reference**. A surface
pressure `p_s(x,y)` is depth-independent, so `∇p_s` drives only the **barotropic** (depth-mean)
flow — it cannot change the baroclinic shear. Decomposing the prognostic overturning
(`experiments/diagnose_amoc_barotropic.py`, WOA18 T31) shows the barrier lives entirely in the
*baroclinic* mode, so a surface-pressure solve cannot touch it:

| quantity | value |
|---|---|
| AMOC upper cell 26.5 N, barotropic **removed** (baroclinic) | **326.2 Sv** |
| AMOC upper cell 26.5 N, barotropic **included** | **326.2 Sv** (identical) |
| global net meridional transport, max\|net/lat\| | **~0 Sv** (the old ±350 Sv spurious mode is gone) |
| depth-integrated `\|div(∫u dz)\|` | **~2e-19** (non-divergent to machine precision) |

The barotropic mode is **already non-divergent and net-zero** — `barotropic.rigid_lid_project`
(`veros_driver.py:219`) eliminated the old ±350 Sv spurious net the roadmap worried about. The
326 Sv AMOC is **pure baroclinic overturning**, identical with or without barotropic removal.

**Conclusion — both candidate levers are now settled, and the barrier is a *resolution* limit.**
GM (~1 %, necessary) and the rigid-lid projection (already done; surface-pressure adds nothing
to the AMOC) are both addressed. The remaining 326 Sv is the T31 **baroclinic geostrophic**
overturning, ~20–40× too large because the grid cannot resolve the eddies that would flatten the
isopycnals. The genuine path forward is **finer resolution** (eddy-permitting, ≪ T31), not a
further T31 parameterization. The shipped **diagnostic + THC** path (which gives a stable,
density-driven ~15 Sv AMOC and the P4 tipping result) remains the correct T31 production choice;
the prognostic-momentum core stays research-in-progress behind its default-off flag, with GM
available as the (now validated) eddy closure for when resolution increases.

**Decision (2026-06-22): ship the working diagnostic + THC AMOC as production; S5 stays
research-in-progress behind the default-off flag.** Rationale: the production path gives a
**stable ~15 Sv AMOC** and already delivered the **P4 bistability/tipping** result, which *is*
density-driven — the THC closure scales with the subpolar−subtropical density contrast, so it
restores the `d(AMOC)/d(density)` pathway that the bare thermal-wind diagnostic lacks. S5 is a
rigor upgrade, **not a blocker** for the forcing-response science or release. **Both candidate
levers are now settled (see below): GM gives ~1 % and the surface-pressure reference is not the
lever (the barotropic mode is already clean) — the barrier is a *resolution* limit, so the next
real S5 increment is finer (eddy-permitting) resolution.** Reproduce: `diagnose_prognostic_amoc.py`
(barrier), `diagnose_gm_amoc.py` (GM), `diagnose_amoc_barotropic.py` (baroclinic/barotropic split).

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
