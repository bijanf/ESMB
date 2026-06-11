"""Validated reference: Google `dinosaur` multi-level dry dycore, Held-Suarez forcing.

WHY THIS EXISTS
---------------
Chronos-ESM's native atmosphere is SINGLE-LEVEL (barotropic): it cannot perform
baroclinic instability, so it has no synoptic systems and no real jets/ITCZ (see
the README "Known limitations"). The plan is to replace it with a MULTI-LEVEL
spectral primitive-equation dycore. We use Google Research's `dinosaur` (the
differentiable JAX dycore behind NeuralGCM) DIRECTLY.

Note: the `jcm` package (a SPEEDY-physics wrapper on dinosaur) is broken on PyPI
(jcm 1.1.1 is built against an unreleased dinosaur and imports `SI_SCALE`, which
exists in no released dinosaur), so we build on `dinosaur` directly instead.

This script is the validated proof that the dycore produces realistic baroclinic
circulation: a dry Held-Suarez (1994) benchmark spins up a mid-latitude westerly
jet (~20-30 m/s in the upper troposphere) and surface trade easterlies from an
isothermal rest state, via genuine baroclinic instability.

    pip install dinosaur-dycore==1.2.1      # compatible with jax 0.8.1; no upgrade
    python experiments/dino_held_suarez.py [days]

It also exposes build_held_suarez_model() for reuse by the coupled integration.
"""

import argparse
import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dinosaur import (coordinate_systems, spherical_harmonic, sigma_coordinates,  # noqa: E402
                      primitive_equations as pe, primitive_equations_states as pes,
                      held_suarez, time_integration as ti, scales, xarray_utils)

units = scales.units


def build_held_suarez_model(truncation="T31", layers=24, dt_minutes=20.0,
                            diffusion_tau_hours=2.0, diffusion_order=2):
    """Build a dinosaur dry primitive-equation model with Held-Suarez forcing.

    Returns a dict with: coords, specs, step_fn (one nondimensional step),
    init_state, steps_per_day, and a u-velocity extractor.
    """
    grid = getattr(spherical_harmonic.Grid, truncation)()
    coords = coordinate_systems.CoordinateSystem(
        horizontal=grid,
        vertical=sigma_coordinates.SigmaCoordinates.equidistant(layers))
    specs = pe.PrimitiveEquationsSpecs.from_si()

    init_fn, aux = pes.isothermal_rest_atmosphere(
        coords, specs, p0=1e5 * units.pascal, p1=5e3 * units.pascal)
    ref_temps = aux[xarray_utils.REF_TEMP_KEY]
    orography_modal = coords.horizontal.to_modal(aux[xarray_utils.OROGRAPHY])

    eq = pe.PrimitiveEquations(ref_temps, orography_modal, coords, specs)
    hs = held_suarez.HeldSuarezForcing(coords, specs, ref_temps)
    ode = ti.compose_equations([eq, hs])  # dycore (ImEx) + forcing (explicit)

    dt = specs.nondimensionalize(dt_minutes * units.minute)
    diff = ti.horizontal_diffusion_step_filter(
        coords.horizontal, dt,
        tau=specs.nondimensionalize(diffusion_tau_hours * units.hour),
        order=diffusion_order)
    step_fn = ti.step_with_filters(ti.imex_rk_sil3(ode, dt), [diff])
    steps_per_day = int(round(specs.nondimensionalize(1 * units.day) / dt))

    sin_lat = np.asarray(coords.horizontal.nodal_mesh[1])      # (nlon, nlat)
    cos_lat = np.sqrt(np.maximum(1 - sin_lat ** 2, 1e-12))

    def zonal_u(state):
        """Zonal-mean zonal wind [m/s], shape (layers, nlat)."""
        d = pe.compute_diagnostic_state(state, coords)
        u = specs.dimensionalize(np.asarray(d.cos_lat_u)[0] / cos_lat[None],
                                 units.meter / units.second).magnitude
        return u.mean(axis=1)  # mean over longitude -> (layers, nlat)

    return dict(coords=coords, specs=specs, step_fn=step_fn,
                init_state=init_fn(rng_key=jax.random.PRNGKey(0)),
                steps_per_day=steps_per_day, zonal_u=zonal_u, ref_temps=ref_temps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="?", type=int, default=60)
    ap.add_argument("--out", default="docs/figures/dino_held_suarez_jet.png")
    args = ap.parse_args()

    m = build_held_suarez_model()
    coords, zonal_u = m["coords"], m["zonal_u"]
    lat = np.rad2deg(np.arcsin(np.asarray(coords.horizontal.nodal_mesh[1][0])))
    sigma = np.asarray(coords.vertical.centers)

    advance = jax.jit(ti.repeated(m["step_fn"], m["steps_per_day"] * 5))  # 5-day chunks
    state = m["init_state"]

    def jetmetrics(state, tag):
        uzm = zonal_u(state)
        mid = uzm[:, (np.abs(lat) > 30) & (np.abs(lat) < 60)].mean()
        upper_mid = uzm[len(sigma) // 4, (np.abs(lat) > 30) & (np.abs(lat) < 60)].mean()
        trop_sfc = uzm[-1, np.abs(lat) < 20].mean()
        print(f"  {tag:7s} |u|max {np.abs(uzm).max():5.1f}  upper-mid jet {upper_mid:+5.1f}  "
              f"sfc trades {trop_sfc:+5.1f} m/s")

    print(f"dinosaur Held-Suarez: T31, {len(sigma)} sigma levels, {args.days} days")
    jetmetrics(state, "day0")
    for d in range(1, args.days // 5 + 1):
        state = advance(state)
        jetmetrics(state, f"day{d*5}")

    # Zonal-mean jet figure (latitude-sigma cross-section).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    uzm = zonal_u(state)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    vlim = float(np.nanpercentile(np.abs(uzm), 99)) or 1.0
    cf = ax.contourf(lat, sigma, uzm, levels=np.linspace(-vlim, vlim, 21), cmap="RdBu_r", extend="both")
    ax.contour(lat, sigma, uzm, levels=[0], colors="k", linewidths=0.4)
    ax.invert_yaxis()
    ax.set_xlabel("Latitude"); ax.set_ylabel("sigma (surface=1)")
    ax.set_title(f"dinosaur Held-Suarez zonal-mean U [m/s], day {args.days}")
    fig.colorbar(cf, ax=ax, label="m/s")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
