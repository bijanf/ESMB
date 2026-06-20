"""Differentiable calibration of the AMOC tipping point (P4).

A tipping threshold is a BIFURCATION (a saddle-node fold), not a smooth output of one
forward run -- so you cannot tune it by backprop-through-the-hosing-sweep (the trajectory
sensitivity diverges *at* the fold, and you'd differentiate through 200+ model-years and a
collapse event). The rigorous differentiable way to tune a fold is FOLD CONTINUATION:
implicit-differentiate the {steady-state, zero-eigenvalue} condition. Here that is cheap
because the THC closure (chronos_esm/ocean/overturning.py) is a Stommel box model in
disguise, so we reduce it, fit its 2 lumped constants to GCM anchors, and AD+Newton-invert
for the parameters that put F_crit at a target (default 0.6 Sv). One confirmatory GCM
hosing run then validates -- replacing a multi-job parameter grid with a single run.

Reduced model (same functional form as the closure):
    s              subpolar-minus-subtropical UPPER-OCEAN salinity contrast [psu] (= dS)
    contrast(s)    = rhoT + g * BETA_S * s          # rhoT = thermal density contrast
    A(s)           = Ceff * kvel_rel * softplus(contrast / DRHO_SCALE)   # AMOC [Sv]
    salt balance   ds/dt = -kappa*F + mu*A(s)*(-s)  # hosing freshens; AMOC imports salt
    steady on-state    kappa*F = mu*A(s)*(-s) =: R(s);  R->0 as s->0- and s->-inf
    => fold  F_crit(theta) = max_s R(s)/kappa   (saddle-node; envelope thm => exact AD)

Usage:
    python experiments/calibrate_amoc_fold.py --target 0.6
    python experiments/calibrate_amoc_fold.py --target 0.6 \
        --anchors /p/tmp/$USER/amoc_hose_cd300_kv6e-5_hg4.0/hysteresis.npz:6e-5:4.0 ...
"""
import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap

jax.config.update("jax_enable_x64", True)

BETA_S, DRHO_SCALE = 0.78, 0.10           # match chronos_esm/ocean/overturning.py
RHO_T = 4.0                                # thermal density contrast ~ alpha*dT (20 K)
S0 = -1.0                                  # WOA subpolar salinity contrast [psu]
KVEL_REF = 1.0e-4                          # reference thc_k_vel (kvel_rel = k_vel/KVEL_REF)
_S_GRID = jnp.linspace(-3.0, -0.02, 4000)
sp = jax.nn.softplus


def amoc(s, g, kvel_rel, Ceff):
    return Ceff * kvel_rel * sp((RHO_T + g * BETA_S * s) / DRHO_SCALE)


def fcrit(g, kvel_rel, Ceff, mk):
    """Fold F_crit [Sv] = max_s mu/kappa * A(s) * (-s)."""
    R = vmap(lambda s: mk * amoc(s, g, kvel_rel, Ceff) * (-s))(_S_GRID)
    return jnp.max(R)


def calibrate(onstate_anchor=(4.0, 1.0, 28.0), marginal_anchor=(4.0, 1.0, 0.9)):
    """Fit the two lumped constants (Ceff, mu/kappa) to GCM anchors.

    onstate_anchor:  (g, kvel_rel, AMOC_Sv at F=0)        -> Ceff
    marginal_anchor: (g, kvel_rel, F_crit_Sv)             -> mu/kappa
    """
    g0, kr0, a0 = onstate_anchor
    Ceff = a0 / (kr0 * float(sp((RHO_T + g0 * BETA_S * S0) / DRHO_SCALE)))
    g1, kr1, f1 = marginal_anchor
    mk = f1 / float(fcrit(g1, kr1, Ceff, 1.0))
    return Ceff, mk


def newton_invert(target, kvel_rel, Ceff, mk, g_init=5.0, n=15):
    """Solve haline_gain such that F_crit = target, via AD-Newton on the fold."""
    g = g_init
    for _ in range(n):
        f = float(fcrit(g, kvel_rel, Ceff, mk)) - target
        df = float(grad(lambda gg: fcrit(gg, kvel_rel, Ceff, mk))(g))
        g = g - f / df
    return g


def _load_anchors(specs):
    """specs: list of 'path.npz:kvel_rel:gain' -> [(g, kvel_rel, onstate, F_crit), ...].

    Reads a run_amoc_hosing hysteresis.npz; on-state = AMOC at F=0 up-leg, and F_crit is
    estimated as the up-leg hosing where AMOC first falls below half the on-state.
    """
    out = []
    for spec in specs:
        path, kr, g = spec.split(":")
        d = np.load(path, allow_pickle=True)
        leg, f, a = d["leg"].astype(str), d["hosing_sv"], d["amoc_sv"]
        up = leg == "up"
        fu, au = f[up], a[up]
        on = float(au[0])
        below = np.where(au < 0.5 * on)[0]
        fcr = float(fu[below[0]]) if below.size else float("nan")
        out.append((float(g), float(kr), on, fcr))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.6, help="target F_crit [Sv]")
    ap.add_argument("--kvel-rel", type=float, default=0.6,
                    help="k_vel / 1e-4 at which to solve for the gain")
    ap.add_argument("--anchors", nargs="*", default=None,
                    help="GCM anchors 'hysteresis.npz:kvel_rel:gain' to recalibrate the box")
    args = ap.parse_args()

    if args.anchors:
        rows = _load_anchors(args.anchors)
        # on-state anchor: the run with the strongest AMOC; marginal anchor: median F_crit
        rows_on = max(rows, key=lambda r: r[2])
        onA = (rows_on[0], rows_on[1], rows_on[2])
        fin = [r for r in rows if np.isfinite(r[3])]
        marg = min(fin, key=lambda r: abs(r[3] - args.target)) if fin else (4.0, 1.0, 0.9)
        margA = (marg[0], marg[1], marg[3])
        print(f"GCM anchors loaded ({len(rows)}): " +
              ", ".join(f"g{r[0]:.0f}/kv{r[1]:.2f}->on{r[2]:.0f}Sv,Fc{r[3]:.2f}" for r in rows))
        Ceff, mk = calibrate(onA, margA)
    else:
        print("no --anchors: using rough hand anchors (28 Sv on-state, ~0.9 Sv marginal)")
        Ceff, mk = calibrate()

    print(f"calibrated:  Ceff={Ceff:.2f} Sv,  mu/kappa={mk:.4f}\n")
    print("F_crit [Sv] over (haline_gain, k_vel/1e-4):")
    print("          g=3    g=4    g=5    g=6    g=7")
    for kr in (1.0, 0.6, 0.4):
        row = [float(fcrit(g, kr, Ceff, mk)) for g in (3., 4., 5., 6., 7.)]
        print(f"  kvel={kr:>3.1f}  " + " ".join(f"{v:6.2f}" for v in row))

    dF_dg = float(grad(lambda g: fcrit(g, args.kvel_rel, Ceff, mk))(5.0))
    dF_dk = float(grad(lambda kr: fcrit(5.0, kr, Ceff, mk))(args.kvel_rel))
    print(f"\nAD fold gradients @ (g=5, kvel={args.kvel_rel}):"
          f"  dFc/dg={dF_dg:+.3f} Sv/unit,  dFc/dkvel={dF_dk:+.3f} Sv/unit")

    g = newton_invert(args.target, args.kvel_rel, Ceff, mk)
    on = float(amoc(S0, g, args.kvel_rel, Ceff))
    print(f"\nNewton(AD) -> haline_gain = {g:.3f} gives F_crit = {args.target:.2f} Sv "
          f"at k_vel = {args.kvel_rel*KVEL_REF:.1e} m/s (on-state {on:.1f} Sv)")
    print(f"Confirm with ONE GCM run:\n  python experiments/run_amoc_hosing.py "
          f"--haline-gain {g:.2f} --k-vel {args.kvel_rel*KVEL_REF:.1e} --contrast-depth 300 "
          f"--fmax 0.9 --nsteps 9 --hold-years 15 --spinup-years 20")


if __name__ == "__main__":
    main()
