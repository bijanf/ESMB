"""Showcase animation: baroclinic eddies emerging from rest in the `dinosaur` dycore.

Chronos-ESM's multi-level atmosphere is Google's differentiable `dinosaur` spectral
primitive-equation core (the dycore behind NeuralGCM). Run with Held-Suarez (1994)
forcing from an isothermal rest state, it spins up a mid-latitude westerly jet and a
train of baroclinic eddies via genuine baroclinic instability -- the textbook proof
that the dynamics are real (see experiments/dino_held_suarez.py, which scores the jet).

This renders that spin-up as a movie: lower-tropospheric relative vorticity on the
global aquaplanet, one frame per model day. Smooth zonal flow -> growing waves ->
fully turbulent synoptic weather. Honest scope: this is the DRY dycore on an
idealized aquaplanet (no continents, Held-Suarez forcing), not the coupled Earth run.

    PYTHONPATH=$PWD JAX_PLATFORMS=cpu venv/bin/python experiments/make_showcase_animation.py
    # options: --days 180 --level-sigma 0.85 --gif docs/figures/dino_baroclinic_vorticity.gif

Outputs an embeddable .gif (subsampled, small) + a full-quality .mp4 (ffmpeg).
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dinosaur import primitive_equations as pe  # noqa: E402,F401
from dinosaur import scales  # noqa: E402
from dinosaur import time_integration as ti  # noqa: E402

from experiments.dino_held_suarez import build_held_suarez_model  # noqa: E402

units = scales.units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180,
                    help="number of model days to animate (1 frame/day)")
    ap.add_argument("--level-sigma", type=float, default=0.85,
                    help="sigma level for the vorticity map (1=surface)")
    ap.add_argument("--gif", default="docs/figures/dino_baroclinic_vorticity.gif")
    ap.add_argument("--mp4", default="docs/figures/dino_baroclinic_vorticity.mp4")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    print(f"Building dinosaur Held-Suarez model (T31, 24 levels)...")
    m = build_held_suarez_model()
    coords = m["coords"]
    sin_lat = np.asarray(coords.horizontal.nodal_mesh[1])  # (nlon, nlat)
    lat = np.rad2deg(np.arcsin(sin_lat[0]))                # (nlat,)
    nlon = sin_lat.shape[0]
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    sigma = np.asarray(coords.vertical.centers)
    level = int(np.argmin(np.abs(sigma - args.level_sigma)))
    print(f"  vorticity at sigma={sigma[level]:.3f} (level {level}/{len(sigma)})")

    specs = m["specs"]
    advance = jax.jit(ti.repeated(m["step_fn"], m["steps_per_day"]))
    state = m["init_state"]

    # --- integrate, storing one vorticity map per day -------------------------
    # NB: the dycore state is NONDIMENSIONAL; dimensionalize vorticity to s^-1
    # (the scaled value labelled as physical would be ~10^3x too large).
    fields = []
    t0 = time.time()
    for day in range(args.days + 1):
        if day > 0:
            state = advance(state)
        vort = np.asarray(coords.horizontal.to_nodal(state.vorticity))  # (lev,nlon,nlat)
        vort = specs.dimensionalize(vort, units.second ** -1).magnitude  # -> s^-1
        fields.append(vort[level].T * 1e5)  # (nlat, nlon), units 1e-5 s^-1
        if day % 20 == 0:
            jax.block_until_ready(state.vorticity)
            print(f"  day {day:4d}/{args.days}  ({time.time()-t0:.0f}s)")
    fields = np.stack(fields)  # (nframes, nlat, nlon)

    # robust symmetric colour limit from the eddy-active second half
    vlim = float(np.nanpercentile(np.abs(fields[len(fields) // 2:]), 99.0))
    print(f"  colour limit +/-{vlim:.1f} x1e-5 s^-1")

    # --- render frames --------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    tmp = os.path.join(os.path.dirname(args.mp4) or ".", "_anim_frames")
    if os.path.isdir(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=True)

    pil_frames = []
    for i, f in enumerate(fields):
        fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=100)
        pcm = ax.pcolormesh(lon, lat, f, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                            shading="auto")
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_yticks([-60, -30, 0, 30, 60])
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
        ax.set_title("Chronos-ESM · dinosaur dycore — relative vorticity "
                     f"(σ≈{sigma[level]:.2f})\nbaroclinic eddies from rest · day {i}",
                     fontsize=9)
        cb = fig.colorbar(pcm, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("ζ  (×10⁻⁵ s⁻¹)", fontsize=8)
        fig.tight_layout()
        png = os.path.join(tmp, f"frame_{i:04d}.png")
        fig.savefig(png)
        plt.close(fig)
        # build a downsampled RGB frame for the GIF (smaller file)
        im = Image.open(png).convert("RGB")
        im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
        pil_frames.append(im)

    # --- GIF (embeddable, subsample every 2nd frame to keep it small) ---------
    os.makedirs(os.path.dirname(args.gif) or ".", exist_ok=True)
    gif_frames = pil_frames[::2]
    gif_frames[0].save(
        args.gif, save_all=True, append_images=gif_frames[1:],
        duration=int(1000 / args.fps), loop=0, optimize=True)
    gif_mb = os.path.getsize(args.gif) / 1e6
    print(f"saved {args.gif}  ({gif_mb:.1f} MB, {len(gif_frames)} frames)")

    # --- MP4 (full quality, via ffmpeg) --------------------------------------
    if shutil.which("ffmpeg"):
        os.makedirs(os.path.dirname(args.mp4) or ".", exist_ok=True)
        cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
               "-i", os.path.join(tmp, "frame_%04d.png"),
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", args.mp4]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            mp4_mb = os.path.getsize(args.mp4) / 1e6
            print(f"saved {args.mp4}  ({mp4_mb:.1f} MB)")
        else:
            print("ffmpeg failed:\n" + r.stderr[-800:])
    else:
        print("ffmpeg not found -- skipped mp4 (gif written)")

    shutil.rmtree(tmp, ignore_errors=True)
    print("done.")


if __name__ == "__main__":
    main()
