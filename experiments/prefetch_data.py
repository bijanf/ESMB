"""Stage all datasets Chronos-ESM needs into the local pooch cache.

Datasets are NOT committed to the repo; they download on first use and cache to
``~/.cache/chronos_esm`` (``pooch.os_cache('chronos_esm')``). On HPC where compute
nodes have NO internet, run this ONCE on a login node (which has internet); the
SLURM job then reads from the shared cache and needs no network.

    python experiments/prefetch_data.py           # fetch WOA18 (~8 MB) + ETOPO1 (~930 MB)
    python experiments/prefetch_data.py --era5    # also fetch ERA5 (needs ~/.cdsapirc)
    python experiments/prefetch_data.py --build   # + build the model once as a full preflight
    python experiments/prefetch_data.py --check    # verify the cache only (no download); exit 1 if missing
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pooch  # noqa: E402

from chronos_esm import data  # noqa: E402

CACHE = str(pooch.os_cache("chronos_esm"))
# files the control run requires (WOA18 ocean IC + ETOPO1 bathymetry/orography)
REQUIRED = ["woa18_decav_t00_5d.nc", "woa18_decav_s00_5d.nc", "etopo1.nc"]


def _human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def _report():
    print(f"\nCache: {CACHE}")
    ok = True
    for f in REQUIRED:
        p = os.path.join(CACHE, f)
        if os.path.exists(p):
            print(f"  [OK]   {f}  ({_human(os.path.getsize(p))})")
        else:
            print(f"  [MISS] {f}")
            ok = False
    return ok


def check():
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(CACHE, f))]
    if missing:
        print(f"Data cache INCOMPLETE in {CACHE}; missing: {missing}")
        print("On a login node with internet, run: python experiments/prefetch_data.py")
        return False
    print(f"Data cache OK ({CACHE}): {', '.join(REQUIRED)} present.")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the cache without downloading; exit 1 if incomplete")
    ap.add_argument("--era5", action="store_true",
                    help="also fetch the ERA5 validation climatology (needs ~/.cdsapirc)")
    ap.add_argument("--build", action="store_true",
                    help="also build init_model + DinoAtmosphere once (full data->model preflight)")
    args = ap.parse_args()

    if args.check:
        sys.exit(0 if check() else 1)

    print(f"pooch cache directory: {CACHE}")
    print("Fetching WOA18 temperature + salinity (5-deg, ~8 MB) ...", flush=True)
    t, s = data.fetch_woa18_temp()
    print(f"  {t}\n  {s}")
    print("Fetching ETOPO1 bathymetry (~930 MB -- this can take a while) ...", flush=True)
    e = data.fetch_etopo1()
    print(f"  {e}")

    if args.era5:
        print("Fetching ERA5 monthly-mean climatology (needs ~/.cdsapirc) ...", flush=True)
        try:
            from chronos_esm.validation import obs
            print(f"  {obs.fetch_era5_climatology()}")
        except Exception as ex:  # noqa: BLE001
            print(f"  ERA5 fetch FAILED ({type(ex).__name__}: {ex}). "
                  "It is only needed for validation/scoring, not the run.")

    if args.build:
        print("Preflight: building init_model(ocean_ic='woa') + DinoAtmosphere() ...", flush=True)
        from chronos_esm import main
        from chronos_esm.atmos.dino_atmos import DinoAtmosphere
        st = main.init_model(ocean_ic="woa")
        main.ocean_masks(nz=st.ocean.u.shape[0])
        DinoAtmosphere()
        print("  model built OK from cached data.")

    if not _report():
        print("\nWARNING: some required datasets are still missing.")
        sys.exit(1)
    print("\nDone. Datasets are cached; compute-node jobs can now read from the cache "
          "with no network. Submit with: sbatch experiments/run_dino_control_slurm.sh --years 100")


if __name__ == "__main__":
    main()
