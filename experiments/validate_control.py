"""
Validate a Chronos-ESM control run against observations (WOA18 + ERA5).

Builds a model climatology (time-mean of saved states, or a short in-process
demo run), scores each prognostic surface field against regridded obs, compares
AMOC to RAPID, and writes a scorecard + Nature-style figures.

Usage:
    # Score saved run output (climatology = mean over matched states):
    python experiments/validate_control.py --states "outputs/control_run/state_*.nc"

    # No long run yet? Exercise the whole pipeline on a short in-process run:
    python experiments/validate_control.py --demo 20

    # Include the atmosphere (downloads/uses cached ERA5 via your ~/.cdsapirc):
    python experiments/validate_control.py --states "..." --era5

Outputs go to outputs/validation/ (scorecard.txt + *.pdf).
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm.validation import obs, scorecard  # noqa: E402


def _to_celsius(field):
    """Auto-convert an ocean temperature field to degC (handles K or degC)."""
    field = np.asarray(field, float)
    return field - 273.15 if np.nanmean(field) > 100.0 else field


def canonical_fields(state):
    """Extract the canonical surface fields (NumPy, model units) from a state."""
    a, o, f = state.atmos, state.ocean, state.fluxes
    # Reduce surface pressure to MEAN SEA LEVEL before comparing to ERA5 MSL: over
    # topography the surface pressure is much lower than the sea-level pressure
    # (~550 hPa over a 5 km plateau), so comparing raw exp(ln_ps) to MSL is wrong
    # over all land. Hypsometric reduction p_msl = p_s * exp(Phi_s / (R_d * T_ref)).
    # Use a FIXED standard-atmosphere reference T (288 K), NOT the instantaneous
    # single-level T: topography is static, so a static scale height removes it
    # cleanly, whereas the model's fluctuating T amplifies variance over high,
    # cold terrain (exp() of a large, noisy exponent) and inflates the MSLP
    # variance far above observed.
    RD = 287.0
    T_REF = 288.0
    phi_s = np.asarray(a.phi_s)
    mslp = np.asarray(np.exp(a.ln_ps)) * np.exp(phi_s / (RD * T_REF))
    return {
        "sst": _to_celsius(np.asarray(o.temp[0])),          # degC
        "sss": np.asarray(o.salt[0]),                       # psu
        "t2m": np.asarray(a.temp),                          # K (near-surface proxy)
        "u_sfc": np.asarray(a.u),                           # m/s
        "v_sfc": np.asarray(a.v),                           # m/s
        "precip": np.asarray(f.precip),                     # kg/m^2/s
        "mslp": mslp,                                        # Pa (reduced to sea level)
    }


def mean_fields(states):
    """Time-mean of canonical fields across a list of states."""
    acc = None
    for st in states:
        cf = canonical_fields(st)
        if acc is None:
            acc = {k: [v] for k, v in cf.items()}
        else:
            for k, v in cf.items():
                acc[k].append(v)
    return {k: np.nanmean(np.stack(v, 0), axis=0) for k, v in acc.items()}


def load_states(pattern):
    from chronos_esm import io
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No files match {pattern!r}")
    print(f"Loading {len(paths)} state file(s) for climatology...")
    return [io.load_state_from_netcdf(p) for p in paths], paths


def demo_states(n):
    """Run n coupled steps from a fresh init and return the per-step states."""
    from chronos_esm import main
    print(f"[demo] running {n} coupled steps from init_model(ocean_ic='woa') ...")
    state = main.init_model(ocean_ic="woa")
    ocean_mask_3d, surface_mask = main.ocean_masks(nz=state.ocean.u.shape[0])
    params = main.ModelParams(mask=surface_mask, ocean_mask_3d=ocean_mask_3d)
    regridder = main.regrid.Regridder()
    states = []
    for i in range(n):
        state = main.step_coupled(state, params, regridder)
        states.append(state)
    return states


def amoc_vs_rapid(ocean_state):
    from chronos_esm.ocean import diagnostics
    d = diagnostics.compute_amoc_diagnostics(ocean_state)
    model_amoc = float(d.get("amoc_max", float("nan")))
    ref = obs.AMOC_RAPID
    return model_amoc, ref, dict((k, float(v)) for k, v in d.items())


def main_cli():
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--states", help="glob of saved state_*.nc files")
    g.add_argument("--demo", type=int, help="run N in-process steps instead")
    ap.add_argument("--era5", action="store_true",
                    help="include atmosphere (download/use cached ERA5)")
    ap.add_argument("--outdir", default="outputs/validation")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    if args.states:
        states, _ = load_states(args.states)
    else:
        states = demo_states(args.demo)

    model_fields = mean_fields(states)

    # --- obs bundles ---
    print("Fetching WOA18 ocean climatology (cached) ...")
    ocean_surface = obs.woa18_surface()

    era5 = None
    if args.era5:
        try:
            print("Fetching ERA5 climatology (this can queue on first call) ...")
            era5 = obs.era5_climatology_fields()
        except Exception as e:  # noqa: BLE001
            print(f"WARNING: ERA5 unavailable ({type(e).__name__}: {e}). "
                  "Scoring ocean only.")
    else:
        print("ERA5 skipped (pass --era5 to include the atmosphere).")

    obs_spec = scorecard.assemble_obs(ocean_surface=ocean_surface, era5=era5)
    model_fields = {k: v for k, v in model_fields.items() if k in obs_spec}

    rows, figs = scorecard.run_scorecard(
        model_fields, obs_spec, outdir=args.outdir,
        make_figures=not args.no_figures)

    table = scorecard.format_scorecard(rows)

    # --- AMOC scalar ---
    m_amoc, ref, amoc_all = amoc_vs_rapid(states[-1].ocean)
    amoc_line = (f"AMOC max = {m_amoc:6.2f} Sv   vs {ref['source']} "
                 f"{ref['value']:.1f} +/- {ref['sd']:.1f} Sv   "
                 f"(diff {m_amoc - ref['value']:+.2f} Sv)")

    report = ("Chronos-ESM control-run validation\n"
              "==================================\n"
              f"climatology from {len(states)} state(s)\n\n"
              + table + "\n\n" + amoc_line + "\n")
    print("\n" + report)

    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "scorecard.txt"), "w") as fh:
        fh.write(report)
    if figs:
        print(f"\nWrote {len(figs)} figure(s) to {args.outdir}/")
    print(f"Scorecard written to {args.outdir}/scorecard.txt")


if __name__ == "__main__":
    main_cli()
