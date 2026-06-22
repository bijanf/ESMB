# Chronos-ESM v0.1.0 — first public research-preview

**Chronos-ESM** is a fully **differentiable**, JAX-based coupled Earth System Model — a
multi-level [`dinosaur`](https://github.com/neuralgcm/dinosaur) spectral atmosphere ↔ z-level
ocean ↔ slab/bucket land ↔ Semtner sea ice — on a T31 (~3.75°) grid, running on GPU/CPU. The
whole coupled step flows through `jax.jit`/`jax.grad`, so `d(climate)/d(parameter)` is a single
`jax.grad` away (the design goal: gradient-based calibration, sensitivity, and data assimilation).

This first public release demonstrates **four working-ESM milestones**, each validated and with
its limitations documented honestly (see the README **Project Status** and **Validation**):

| Milestone | Result |
|---|---|
| **Active multi-level atmosphere** | the `dinosaur` dycore is the coupled atmosphere, end-to-end differentiable |
| **CO₂ forcing response** | free-ocean abrupt-2×CO₂ → **+1.58 K** (TCR-like proxy) |
| **AMOC tipping** | verified saddle-node hysteresis (~**[0.38, 0.75] Sv** hosing) via a density-driven thermohaline closure |
| **Paleo (mid-Holocene 6 ka)** | real seasonal cycle + orbital forcing → **NH summer warming +1.1 K**, monsoon intensification (Asia +31%, ITCZ +0.2° N) |

### Honest limitations
- The atmosphere is **single-humidity at T31** → a **weak ITCZ** (precipitation too dry; no
  "Green Sahara" in the paleo run; modest monsoon magnitudes).
- The **AMOC** uses a thermohaline *closure*; the fully prognostic-momentum ocean core hits a
  T31 geostrophic-resolution barrier and remains research-in-progress.
- CO₂ warming is a **transient surface-forcing proxy**, not equilibrium climate sensitivity.

### Getting started
```bash
git clone https://github.com/bijanf/ESMB.git && cd ESMB
pip install -e . && pip install "jax[cpu]"
pytest -q
```
See the README for the validation dashboards and the book-length manual in `docs/manual/`
(`main.pdf`). Licensed **Apache-2.0**. Contributions welcome (`CONTRIBUTING.md`).
