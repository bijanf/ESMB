"""Utility functions for ocean model, including differentiable alternatives."""

import jax.numpy as jnp


def soft_clip(x, lo, hi, sharpness=3.0):
    """Differentiable alternative to jnp.clip.

    Uses tanh to smoothly constrain values to [lo, hi] range.
    At sharpness=10, behavior is nearly identical to hard clip
    within the normal range but has non-zero gradients at boundaries.

    Args:
        x: Input array.
        lo: Lower bound.
        hi: Upper bound.
        sharpness: Controls transition steepness (higher = sharper).

    Returns:
        Array with values smoothly constrained to [lo, hi].
    """
    mid = (lo + hi) / 2.0
    half_range = (hi - lo) / 2.0
    return mid + half_range * jnp.tanh(sharpness * (x - mid) / half_range)
