"""
Loss functions for proxy data assimilation.
"""

import jax
import jax.numpy as jnp

def mse_loss(
    modeled: jnp.ndarray,
    observed: jnp.ndarray,
    mask: jnp.ndarray = None
) -> jnp.ndarray:
    """
    Mean Squared Error loss.
    
    Args:
        modeled: Modeled values (same shape as observed or masked)
        observed: Observed values
        mask: Boolean mask of valid observations
        
    Returns:
        loss: Scalar MSE
    """
    diff = modeled - observed
    
    if mask is not None:
        # Apply mask
        diff = jnp.where(mask, diff, 0.0)
        count = jnp.sum(mask)
        return jnp.sum(diff**2) / jnp.maximum(count, 1.0)
    else:
        return jnp.mean(diff**2)


def climatology_loss(
    state_field: jnp.ndarray,
    climatology_field: jnp.ndarray,
    weight: float = 1.0
) -> jnp.ndarray:
    """
    Regularization loss penalizing deviation from climatology.
    """
    return weight * jnp.mean((state_field - climatology_field)**2)
