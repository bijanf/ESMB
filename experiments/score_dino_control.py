"""Convert dinosaur coupled control checkpoints (.npz) into dashboard-scoreable .nc.

The dinosaur coupled control harness (``experiments/run_dino_control.py`` ->
``chronos_esm.coupler.dino_step``) checkpoints a ``DinoCoupledState`` to a pair of
``.npz`` files (``state_d<DAY>.npz`` for ocean/land/ice + ``state_d<DAY>_dino.npz``
for the dinosaur MODAL atmosphere: vorticity/divergence/temperature_variation/
log_surface_pressure/specific_humidity). The validation dashboard
(``experiments/make_readme_figures.py``) instead reads ``.nc`` files via
``chronos_esm.io.load_state_from_netcdf`` and scores a ``CoupledState`` whose
``atmos`` carries SURFACE fields. There was no route from one to the other.

This is that bridge. For each selected ``.npz`` checkpoint it:

  1. loads the ``DinoCoupledState`` (``dino_step.load_state``);
  2. computes the dinosaur atmosphere SURFACE diagnostics on the dinosaur Gaussian
     grid and regrids them to the linear (ocean) grid -- this REUSES the library's
     own differentiable diagnostic + regridder path
     (``DinoCoupledModel.diagnostics_lin`` ->
     ``dino_coupling.dino_diagnostics_jax`` + ``gauss_to_lin``), which is the same
     physics ``benchmark_dino.py`` extracts via ``DinoAtmosphere.diagnostics`` +
     its ``gauss_to_lin`` (Gaussian->linear lat interp, longitude index-preserving);
  3. assembles a ``coupler.state.CoupledState`` carrying exactly the fields the
     dashboard scores:
        ocean.temp / ocean.salt   <- the checkpoint's ocean (already on the 48x96 grid)
        atmos.temp                <- regridded near-surface air temperature [K]
        atmos.u / atmos.v         <- regridded surface winds [m/s]
        atmos.ln_ps               <- log(regridded MSLP [Pa])
        atmos.phi_s               <- 0 (documented convention: MSLP already reduced
                                        to sea level, so canonical_fields' hypsometric
                                        term exp(phi_s/(Rd*T_ref)) must be 1)
        fluxes.precip             <- regridded precipitation [kg/m^2/s]
  4. saves it to ``<out-dir>/score_d<DAY>.nc`` via ``io.save_state_to_netcdf``.

The dashboard averages a glob of ``.nc`` into a climatology, so one ``.nc`` per
checkpoint is correct (do NOT pre-average here).

Usage:
    python experiments/score_dino_control.py \
        --ctrl-dir outputs/dino_control --last 30 --out-dir outputs/dino_score
    python experiments/make_readme_figures.py 'outputs/dino_score/score_d*.nc' \
        --label 'dino control'

Smoke test (CPU, no cluster data; builds a fresh WOA-init state, no temp left behind):
    PYTHONPATH=$PWD JAX_PLATFORMS=cpu python experiments/score_dino_control.py --smoke-test
"""

import argparse
import glob
import os
import re
import sys
import tempfile

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm import io  # noqa: E402
from chronos_esm.coupler import state as cstate  # noqa: E402
from chronos_esm.coupler import dino_step  # noqa: E402


_DAY_RE = re.compile(r"state_d(\d+)\.npz$")


def _checkpoint_day(path):
    """Absolute simulation day parsed from a ``state_d<DAY>.npz`` filename."""
    m = _DAY_RE.search(os.path.basename(path))
    if m is None:
        return None
    return int(m.group(1))


def _build_coupled_state(model, cs):
    """Build a dashboard-scoreable ``coupler.state.CoupledState`` from a
    ``DinoCoupledState`` ``cs`` using the model's own diagnostics/regridders.

    Only the fields the dashboard scores are physical; the remaining slots that
    ``io.save_state_to_netcdf`` writes (q, co2, vorticity, divergence, psi, chi,
    wind stress, land/ice diagnostics, net fluxes) are filled from the checkpoint
    or with neutral placeholders -- they are NOT read by ``canonical_fields``.
    """
    # Surface diagnostics on the linear (48x96) grid: u_sfc, v_sfc, t_sfc, q_sfc,
    # precip, mslp -- the SAME extraction benchmark_dino does, via the library path.
    diag = model.diagnostics_lin(cs)
    u_sfc = jnp.asarray(diag["u_sfc"])
    v_sfc = jnp.asarray(diag["v_sfc"])
    t_air = jnp.asarray(diag["t_sfc"])     # near-surface air temperature [K]
    q_air = jnp.maximum(jnp.asarray(diag["q_sfc"]), 0.0)
    precip = jnp.maximum(jnp.asarray(diag["precip"]), 0.0)
    mslp = jnp.asarray(diag["mslp"])       # Pa (reduced to sea level)

    ny, nx = t_air.shape
    zeros = jnp.zeros((ny, nx))

    # Atmosphere surface state. phi_s = 0 so canonical_fields' mslp =
    # exp(ln_ps) * exp(0/(Rd*T_ref)) = exp(ln_ps) = mslp (already MSL-reduced).
    atmos = cstate.AtmosState(
        vorticity=zeros, divergence=zeros, temp=t_air,
        ln_ps=jnp.log(jnp.maximum(mslp, 1.0)), q=q_air,
        co2=jnp.ones((ny, nx)) * 280.0,
        u=u_sfc, v=v_sfc, psi=zeros, chi=zeros, phi_s=zeros)

    # Ocean: take temp/salt (and the rest) straight from the checkpoint. The
    # DinoCoupledState ocean is a veros OceanState with all fields populated.
    ocean = cs.ocean

    fluxes = cstate.FluxState(
        net_heat_flux=zeros, freshwater_flux=zeros,
        wind_stress_x=zeros, wind_stress_y=zeros, precip=precip,
        sst=jnp.asarray(ocean.temp[0]),
        carbon_flux_ocean=zeros, carbon_flux_land=zeros)

    return cstate.CoupledState(
        ocean=ocean, atmos=atmos, ice=cs.ice, land=cs.land,
        fluxes=fluxes, time=float(cs.day) * 86400.0)


def convert(ctrl_dir, last, out_dir, model=None):
    """Convert the ``--last`` most-recent ``state_d*.npz`` checkpoints in
    ``ctrl_dir`` to ``score_d<DAY>.nc`` in ``out_dir``. Returns the list of
    written paths. ``model`` may be passed to reuse a built DinoCoupledModel
    (e.g. the smoke test); otherwise one is built here."""
    paths = glob.glob(os.path.join(ctrl_dir, "state_d*.npz"))
    # exclude the *_dino.npz companion files (load_state reconstructs them).
    paths = [p for p in paths if not p.endswith("_dino.npz")]
    dated = [(d, p) for p in paths if (d := _checkpoint_day(p)) is not None]
    if not dated:
        raise SystemExit(f"No state_d*.npz checkpoints found in {ctrl_dir!r}")
    dated.sort(key=lambda dp: dp[0])               # ascending by day
    selected = dated[-last:] if last > 0 else dated
    print(f"Found {len(dated)} checkpoint(s) in {ctrl_dir}; converting the "
          f"{len(selected)} most recent (days {selected[0][0]}..{selected[-1][0]}).")

    if model is None:
        # ocean_ic='woa' matches the control harness init; only the GRID/atmosphere
        # object is used here (the ocean fields come from each checkpoint).
        print("Building DinoCoupledModel(ocean_ic='woa') (grid + atmosphere object) ...")
        model = dino_step.DinoCoupledModel(ocean_ic="woa")

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for day, path in selected:
        cs = dino_step.load_state(path[: -len(".npz")])  # load_state takes path_base
        coupled = _build_coupled_state(model, cs)
        out = os.path.join(out_dir, f"score_d{day}.nc")
        io.save_state_to_netcdf(coupled, out)            # prints "Saved state to ..."
        print(f"  d{day:>6} -> {out}")
        written.append(out)
    return written


# --------------------------------------------------------------------------- #
# Smoke test: build a state, checkpoint it to a temp dir, convert, read back,
# assert the dashboard-scored fields are present/finite/plausible. No cluster
# data and no leftover temp files.
# --------------------------------------------------------------------------- #
def _smoke_test():
    print("[smoke] building DinoCoupledModel(ocean_ic='woa') ...")
    model = dino_step.DinoCoupledModel(ocean_ic="woa")
    cs = model.init_state()
    print("[smoke] stepping one coupling interval (step_fast) ...")
    cs = model.step_fast(cs)

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        ctrl = os.path.join(tmp, "ctrl")
        out = os.path.join(tmp, "out")
        os.makedirs(ctrl, exist_ok=True)
        # Write a checkpoint exactly as the harness does.
        day = int(round(float(cs.day)))
        dino_step.save_state(cs, os.path.join(ctrl, f"state_d{day}"))

        written = convert(ctrl, last=1, out_dir=out, model=model)
        assert len(written) == 1, f"expected 1 .nc, got {len(written)}"
        nc = written[0]
        assert os.path.exists(nc), f"{nc} not written"
        print(f"[smoke] wrote {nc}; reading back via io.load_state_from_netcdf ...")

        st = io.load_state_from_netcdf(nc)

        checks = {
            "atmos.temp":   (np.asarray(st.atmos.temp),   (200.0, 330.0)),
            "atmos.u":      (np.asarray(st.atmos.u),      (-80.0, 80.0)),
            "atmos.v":      (np.asarray(st.atmos.v),      (-80.0, 80.0)),
            "atmos.ln_ps":  (np.asarray(st.atmos.ln_ps),  (np.log(3e4), np.log(1.1e5))),
            "atmos.phi_s":  (np.asarray(st.atmos.phi_s),  (0.0, 0.0)),
            "fluxes.precip":(np.asarray(st.fluxes.precip),(0.0, 1e-2)),
            "ocean.temp":   (np.asarray(st.ocean.temp),   (250.0, 320.0)),
            "ocean.salt":   (np.asarray(st.ocean.salt),   (20.0, 42.0)),
        }
        print("\n[smoke] field | shape | min | max | mean | expect-range | finite?")
        for name, (arr, (lo, hi)) in checks.items():
            finite = bool(np.all(np.isfinite(arr)))
            amin, amax, amean = float(np.min(arr)), float(np.max(arr)), float(np.mean(arr))
            in_range = lo <= amin and amax <= hi if lo != hi else (amin == lo and amax == hi)
            status = "OK" if (finite and in_range) else "FAIL"
            if not (finite and in_range):
                ok = False
            print(f"  {name:<14} {str(arr.shape):<12} {amin:>10.4g} {amax:>10.4g} "
                  f"{amean:>10.4g}  [{lo:g},{hi:g}]  finite={finite}  {status}")

        # Cross-check: the reconstructed MSLP (canonical_fields) round-trips to mslp.
        from experiments.validate_control import canonical_fields
        cf = canonical_fields(st)
        mslp = np.asarray(cf["mslp"])
        print(f"\n[smoke] canonical_fields mslp [Pa]: min={mslp.min():.0f} "
              f"max={mslp.max():.0f} mean={mslp.mean():.0f} finite={np.all(np.isfinite(mslp))}")
        if not (np.all(np.isfinite(mslp)) and 3e4 <= mslp.min() and mslp.max() <= 1.1e5):
            ok = False
            print("[smoke] FAIL: reconstructed MSLP out of range.")

    print(f"\n[smoke] {'PASS' if ok else 'FAIL'} (temp files cleaned up).")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ctrl-dir", help="directory containing state_d*.npz checkpoints")
    ap.add_argument("--last", type=int, default=30,
                    help="convert the N most-recent checkpoints by day (default 30; "
                         "<=0 means all)")
    ap.add_argument("--out-dir", help="directory to write score_d<DAY>.nc files")
    ap.add_argument("--smoke-test", action="store_true",
                    help="run the self-contained CPU smoke test and exit")
    args = ap.parse_args()

    if args.smoke_test:
        raise SystemExit(_smoke_test())

    if not args.ctrl_dir or not args.out_dir:
        ap.error("--ctrl-dir and --out-dir are required (or pass --smoke-test)")

    written = convert(args.ctrl_dir, args.last, args.out_dir)
    print(f"\nWrote {len(written)} .nc file(s) to {args.out_dir}/.")
    print("Score them with:\n  python experiments/make_readme_figures.py "
          f"'{os.path.join(args.out_dir, 'score_d*.nc')}' --label 'dino control'")


if __name__ == "__main__":
    main()
