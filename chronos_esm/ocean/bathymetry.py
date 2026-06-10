"""
Bathymetry helpers: turn a depth field H(x,y) into the static 3-D wet masks the
ocean step uses to enforce solid-boundary no-flux at coasts and the sea floor.

Full-cell ("staircase") representation: a layer k is wet in a column if the top
of the layer lies above the sea floor (z_top[k] < H). Land columns (H = 0) are
fully dry. This is the recognised legitimate baseline (Adcroft, Hill & Marshall
1997); partial bottom cells (fractional thickness) are a later refinement that
reuses the same masked-flux machinery.

All masks are STATIC (computed once from a fixed H), so they are plain constant
arrays used in elementwise multiplies downstream — reverse-mode-autodiff safe.
"""

import jax.numpy as jnp
import numpy as np


def layer_tops(dz):
    """Depth of the TOP interface of each layer [m] (z_top[0] = 0)."""
    dz = np.asarray(dz)
    return np.concatenate([[0.0], np.cumsum(dz)[:-1]])


def build_wet_mask(H, dz):
    """3-D wet center mask maskC[k,y,x] (1.0 = ocean cell, 0.0 = land/below floor).

    Cell (k) is wet where its top interface is above the floor: z_top[k] < H.
    """
    H = np.asarray(H)
    z_top = layer_tops(dz)[:, None, None]  # (nz,1,1)
    maskC = (z_top < H[None, :, :]).astype("float32")
    return jnp.asarray(maskC)


def kbot(maskC):
    """Number of wet levels per column (0 = land) — for diagnostics/plots."""
    return np.asarray(maskC).sum(axis=0).astype(int)


def face_masks(maskC):
    """Build C-grid face masks from the center mask (MITgcm hFacW/S = MIN rule).

    A face is open (1.0) only if BOTH adjacent center cells are wet, so no flux is
    ever exchanged with a dry cell.

    Returns dict with:
      west  : (nz,ny,nx) open area of the face between (i-1) and (i)   [lon axis=2]
      east  : between (i) and (i+1)
      south : between (j-1) and (j)                                    [lat axis=1]
      north : between (j) and (j+1)
      zlow  : (nz,ny,nx) open area of the interface BELOW cell k (between k and k+1)
    Longitude is periodic; latitude is not (north/south are zeroed at the polar
    boundary rows so nothing crosses the pole).
    """
    m = maskC
    west = m * jnp.roll(m, 1, axis=2)
    east = m * jnp.roll(m, -1, axis=2)
    south = m * jnp.roll(m, 1, axis=1)
    north = m * jnp.roll(m, -1, axis=1)
    # Non-periodic latitude: no flux across the south edge of row 0 / north edge of row -1.
    south = south.at[:, 0, :].set(0.0)
    north = north.at[:, -1, :].set(0.0)
    # Vertical interface below cell k is open only if k+1 is also wet (floor = no flux).
    m_below = jnp.concatenate([m[1:], jnp.zeros_like(m[:1])], axis=0)
    zlow = m * m_below
    return {"west": west, "east": east, "south": south, "north": north, "zlow": zlow}
