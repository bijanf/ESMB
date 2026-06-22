"""Mid-Holocene (6 ka) paleo experiment on the FREE seasonal coupled model — P5.

The deliverable of P5: run the freed (q-flux) coupled model with a real seasonal cycle
(chronos_esm.orbital) under a chosen orbit and accumulate the JJA / DJF seasonal-mean
climatology, so a 6 ka run vs a present-day (PI) run isolates the mid-Holocene signal:
the enhanced / northward-shifted NH summer monsoon ("Green Sahara"), driven by the +20-30
W/m^2 boreal-summer insolation anomaly (the orbital forcing is validated in
tests/test_orbital.py).

Run the SAME q-flux + start state for both orbits (only the orbital config differs):

    # present-day control:
    python experiments/run_paleo_midholocene.py --ckpt /p/tmp/.../state_dNNNNNN \
        --orbit pi  --years 40 --clim-start-year 15 --outdir outputs/paleo_pi
    # mid-Holocene:
    python experiments/run_paleo_midholocene.py --ckpt /p/tmp/.../state_dNNNNNN \
        --orbit 6ka --years 40 --clim-start-year 15 --outdir outputs/paleo_6ka

The q-flux comes from the SEASONAL PI control (run_dino_control.py --seasonal). Both runs
use the PI q-flux + start state so the only difference is the orbit (PMIP-style: the q-flux
is the model's climatological flux correction, held fixed across the paleo perturbation).

Writes <outdir>/clim_<orbit>.npz: jja_/djf_ mean of {sst, precip, t2m, u_sfc, v_sfc, mslp}
+ sample counts, plus a resumable state checkpoint. Compare with plot_paleo_midholocene.py.
"""
import argparse
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronos_esm.coupler.dino_step import DinoCoupledModel, load_state, save_state  # noqa: E402
from chronos_esm.ocean.diagnostics import compute_amoc  # noqa: E402
from chronos_esm.config import OCEAN_GRID  # noqa: E402
from chronos_esm import orbital  # noqa: E402

DAYS_PER_YEAR = 365
# JJA = Jun 1 - Aug 31, DJF = Dec 1 - Feb 28 (day 80 = vernal equinox -> day-of-year is the
# real calendar). A boolean season test on day-of-year.
def _is_jja(doy):
    return 151 <= doy <= 243
def _is_djf(doy):
    return doy >= 335 or doy < 59


FIELDS = ("sst", "precip", "t2m", "u_sfc", "v_sfc", "mslp")


def _fields(model, cstate):
    diag = model.diagnostics_lin(cstate)
    return {
        "sst": np.asarray(cstate.ocean.temp[0]) - 273.15,
        "precip": np.maximum(np.asarray(diag["precip"]), 0.0),
        "t2m": np.asarray(diag["t_sfc"]) - 273.15,
        "u_sfc": np.asarray(diag["u_sfc"]),
        "v_sfc": np.asarray(diag["v_sfc"]),
        "mslp": np.asarray(diag["mslp"]),
    }


def main_cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="seasonal-control checkpoint base (no ext)")
    ap.add_argument("--qflux", default=None, help="q-flux .npy (default <ckpt>_qflux.npy)")
    ap.add_argument("--orbit", choices=["pi", "6ka"], default="pi")
    ap.add_argument("--years", type=float, default=40.0)
    ap.add_argument("--clim-start-year", type=float, default=15.0,
                    help="start accumulating the seasonal climatology after this many years "
                         "(let the free model adjust to the orbit first)")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--restore-tau-days", type=float, default=3650.0,
                    help="weak anomaly-restoring timescale for FREE mode")
    ap.add_argument("--outdir", default="outputs/paleo_run")
    ap.add_argument("--resume", action="store_true", help="resume from <outdir>/state_<orbit>")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    qpath = args.qflux or (args.ckpt + "_qflux.npy")
    q_flux = jnp.asarray(np.load(qpath))
    orbit = {"pi": orbital.ORBIT_PI, "6ka": orbital.ORBIT_6KA}[args.orbit]
    model = DinoCoupledModel(q_flux=q_flux, restore_tau_days=args.restore_tau_days,
                             interval=args.interval, seasonal=True, orbit=orbit)
    omask = model.omask

    def sst_mean_C(cs):
        sst = cs.ocean.temp[0]
        return float(jnp.sum(jnp.where(omask, sst, 0.0)) / jnp.sum(omask)) - 273.15

    # --- accumulators (seasonal climatology) ---
    ny, nx = OCEAN_GRID.nlat, OCEAN_GRID.nlon
    acc = {f"jja_{f}": np.zeros((ny, nx)) for f in FIELDS}
    acc.update({f"djf_{f}": np.zeros((ny, nx)) for f in FIELDS})
    n_jja = n_djf = 0
    statebase = os.path.join(args.outdir, f"state_{args.orbit}")
    climpath = os.path.join(args.outdir, f"clim_{args.orbit}.npz")

    if args.resume and os.path.exists(statebase + ".npz"):
        cstate = load_state(statebase)
        if os.path.exists(climpath):
            d = np.load(climpath)
            for k in acc:
                acc[k] = np.asarray(d[k])
            n_jja, n_djf = int(d["n_jja"]), int(d["n_djf"])
        start_day = int(round(cstate.day))
        print(f"RESUME {args.orbit}: day {start_day}, JJA n={n_jja} DJF n={n_djf}", flush=True)
    else:
        cstate = load_state(args.ckpt)
        start_day = int(round(cstate.day))
        print(f"START {args.orbit} from {args.ckpt} (day {start_day}); q-flux {qpath} "
              f"mean {float(jnp.mean(q_flux)):.2f} W/m2; orbit obliquity "
              f"{orbit.obliquity_deg} ecc {orbit.eccentricity} peri {orbit.long_perihelion_deg}",
              flush=True)

    end_day = start_day + int(round(args.years * DAYS_PER_YEAR))
    clim_start_day = start_day + int(round(args.clim_start_year * DAYS_PER_YEAR))
    n_intervals = int(round((end_day - cstate.day) / args.interval))

    for it in range(1, n_intervals + 1):
        cstate = model.step_fast(cstate, co2_ppm=280.0)
        day = int(round(cstate.day))
        if day >= clim_start_day:
            doy = day % DAYS_PER_YEAR
            fl = _fields(model, cstate) if (_is_jja(doy) or _is_djf(doy)) else None
            if fl is not None and _is_jja(doy):
                for f in FIELDS:
                    acc[f"jja_{f}"] += fl[f]
                n_jja += 1
            elif fl is not None and _is_djf(doy):
                for f in FIELDS:
                    acc[f"djf_{f}"] += fl[f]
                n_djf += 1
        if (day % 30 == 0) or (it == n_intervals):
            amoc = float(compute_amoc(cstate.ocean, ocean_mask=omask)["upper_cell_26N"])
            finite = bool(np.isfinite(np.asarray(cstate.ocean.temp)).all())
            print(f"  {args.orbit} day {day:6d} ({day/DAYS_PER_YEAR:6.2f} yr): "
                  f"SST {sst_mean_C(cstate):5.2f}C  AMOC {amoc:5.1f}Sv  "
                  f"JJA n={n_jja} DJF n={n_djf}  finite {finite}", flush=True)
            if not finite:
                save_state(cstate, statebase + "_NAN")
                raise FloatingPointError(f"non-finite at day {day}")
        if (day % DAYS_PER_YEAR == 0) or (it == n_intervals):
            save_state(cstate, statebase)
            # store raw SUMS + counts (exact resume); the plot computes mean = sum/count.
            np.savez(climpath, n_jja=n_jja, n_djf=n_djf, **acc)

    print(f"done {args.orbit}: ran to day {int(round(cstate.day))}; "
          f"JJA samples {n_jja}, DJF samples {n_djf}; climatology -> {climpath}", flush=True)


if __name__ == "__main__":
    main_cli()
