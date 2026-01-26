"""
Differentiable mixing schemes for the Ocean component.

Implements:
1. Isopycnal diffusion (Redi)
2. Gent-McWilliams (GM) parameterization
3. Differentiable slope limiters
"""

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

# Constants
SMALL_SLOPE = 1e-8
MAX_SLOPE = 1e-2


def compute_isopycnal_slopes(
    rho: jnp.ndarray, dx: float, dy: float, dz: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute isopycnal slopes S_x = - (∂ρ/∂x) / (∂ρ/∂z) and S_y = - (∂ρ/∂y) / (∂ρ/∂z).

    Args:
        rho: Density field (nz, ny, nx)
        dx: Grid spacing in x
        dy: Grid spacing in y
        dz: Grid spacing in z (1D array of length nz)

    Returns:
        sx, sy: Slopes in x and y directions
    """
    nz, ny, nx = rho.shape

    # Vertical gradient ∂ρ/∂z (centered difference)
    # Note: z index increases downwards usually, so check sign convention.
    # Assuming index 0 is surface, index nz-1 is bottom.
    # d_rho[k] = rho[k+1] - rho[k]
    # dz[k] is thickness of layer k. Distance between centers is 0.5*(dz[k] + dz[k+1])

    # Simple finite difference for gradients
    # Use jnp.roll for periodic/boundary handling (simplified for T63)

    # x-gradient
    drho_dx = (jnp.roll(rho, -1, axis=2) - jnp.roll(rho, 1, axis=2)) / (2 * dx)

    # y-gradient
    drho_dy = (jnp.roll(rho, -1, axis=1) - jnp.roll(rho, 1, axis=1)) / (2 * dy)

    # z-gradient
    # Use centered differences for interior, one-sided for boundaries

    # Reshape dz for broadcasting: (nz, 1, 1)
    dz_3d = dz.reshape(-1, 1, 1)

    # Distance between cell centers
    # dist[k] = distance between center k and k+1 = 0.5*(dz[k] + dz[k+1])
    dist_interfaces = 0.5 * (dz_3d[:-1] + dz_3d[1:])

    # Gradients at interfaces k+1/2: (rho[k+1] - rho[k]) / dist[k]
    grad_interfaces = (rho[1:] - rho[:-1]) / dist_interfaces

    # Interpolate to cell centers
    # Interior k=1..nz-2: average of grad at k-1/2 and k+1/2
    # Note: grad_interfaces[k] is at k+1/2
    # So grad_center[k] = 0.5 * (grad_interfaces[k-1] + grad_interfaces[k])
    # But we need to weight by layer thickness if variable dz, but simple average is fine for T63

    drho_dz_interior = 0.5 * (grad_interfaces[:-1] + grad_interfaces[1:])

    # Boundaries
    drho_dz_top = grad_interfaces[:1]  # Use forward diff at surface
    drho_dz_bot = grad_interfaces[-1:]  # Use backward diff at bottom

    drho_dz = jnp.concatenate([drho_dz_top, drho_dz_interior, drho_dz_bot], axis=0)

    # Avoid division by zero and ensure stable stratification assumption for slope
    # In stable ocean, rho increases with depth, so drho_dz > 0 (if z increases downwards)
    # Add small epsilon and sign check
    drho_dz_stable = jnp.where(
        jnp.abs(drho_dz) < SMALL_SLOPE, jnp.sign(drho_dz + 1e-16) * SMALL_SLOPE, drho_dz
    )

    sx = -drho_dx / drho_dz_stable
    sy = -drho_dy / drho_dz_stable

    return sx, sy


def slope_limiter(slope: jnp.ndarray, max_slope: float = MAX_SLOPE) -> jnp.ndarray:
    """
    Differentiable slope limiter.

    Instead of hard clipping (which has zero gradient outside bounds),
    use a smooth approximation like softsign or tanh, or a differentiable
    polynomial approximation.

    Here we use a smooth soft-clipping function:
    f(x) = x / (1 + (x/max)^2)^0.5
    This asymptotes to max_slope.
    """
    return slope / jnp.sqrt(1.0 + (slope / max_slope) ** 2)


def compute_gm_bolus_velocity(
    kappa_gm: float, sx: jnp.ndarray, sy: jnp.ndarray, dz: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute Gent-McWilliams bolus velocities.

    u* = ∂_z (κ S_x)
    v* = ∂_z (κ S_y)
    w* = - (∂_x (κ S_x) + ∂_y (κ S_y))

    Args:
        kappa_gm: GM diffusivity coefficient
        sx, sy: Isopycnal slopes
        dz: Vertical grid spacing

    Returns:
        u_bolus, v_bolus, w_bolus
    """
    nz, ny, nx = sx.shape
    dz_3d = dz.reshape(-1, 1, 1)

    # Apply limiter
    sx_lim = slope_limiter(sx)
    sy_lim = slope_limiter(sy)

    # Fluxes Fx = kappa * Sx, Fy = kappa * Sy
    fx = kappa_gm * sx_lim
    fy = kappa_gm * sy_lim

    # Vertical derivatives for u*, v*
    # u* = ∂(fx)/∂z
    # Using centered differences on vertical interfaces

    # Values at interfaces (linear interp)
    fx_face = 0.5 * (fx[:-1] + fx[1:])
    fy_face = 0.5 * (fy[:-1] + fy[1:])

    # Pad with zeros at surface and bottom (no flux through boundaries)
    fx_face = jnp.concatenate(
        [jnp.zeros((1, ny, nx)), fx_face, jnp.zeros((1, ny, nx))], axis=0
    )
    fy_face = jnp.concatenate(
        [jnp.zeros((1, ny, nx)), fy_face, jnp.zeros((1, ny, nx))], axis=0
    )

    # Derivative at cell center
    u_bolus = (fx_face[1:] - fx_face[:-1]) / dz_3d
    v_bolus = (fy_face[1:] - fy_face[:-1]) / dz_3d

    # w* calculation requires horizontal divergence
    # For T63, we assume dx, dy are constant for simplicity here,
    # but in full model they vary with latitude.
    # w* = -∇_h · (F)
    # We won't compute w* explicitly here as it's usually computed
    # via continuity from u* and v* in the main stepper to ensure conservation.
    # Returning zeros for now or implementing if needed.
    w_bolus = jnp.zeros_like(sx)

    return u_bolus, v_bolus, w_bolus


def rotate_tensor(kappa_iso: float, sx: jnp.ndarray, sy: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the small-slope diffusion tensor K.

    K = κ_iso / (1 + S^2) * [
        [1, 0, Sx],
        [0, 1, Sy],
        [Sx, Sy, S^2]
    ]

    Returns:
        K tensor of shape (3, 3, nz, ny, nx)
    """
    sx_lim = slope_limiter(sx)
    sy_lim = slope_limiter(sy)
    s2 = sx_lim**2 + sy_lim**2

    factor = kappa_iso / (1.0 + s2)

    # Construct tensor components
    k00 = factor
    k11 = factor
    k22 = factor * s2
    k02 = factor * sx_lim
    k12 = factor * sy_lim

    # Fill tensor (symmetric)
    # Using a list of arrays then stacking is efficient in JAX
    zeros = jnp.zeros_like(factor)

    row0 = jnp.stack([k00, zeros, k02], axis=0)
    row1 = jnp.stack([zeros, k11, k12], axis=0)
    row2 = jnp.stack([k02, k12, k22], axis=0)

    K = jnp.stack([row0, row1, row2], axis=0)


def compute_vertical_diffusivity(
    rho: jnp.ndarray,
    dz: jnp.ndarray,
    kappa_bg: float = 1e-5,
    kappa_convect: float = 10.0,
) -> jnp.ndarray:
    """
    Compute vertical diffusivity based on static stability.
    
    If stratification is unstable (drho/dz < 0), use large convective diffusivity.
    Otherwise use background diffusivity.
    
    Args:
        rho: Density (nz, ny, nx)
        dz: Layer thicknesses (nz,)
        kappa_bg: Background diffusivity [m^2/s]
        kappa_convect: Convective diffusivity [m^2/s]
        
    Returns:
        kappa_z: Vertical diffusivity at cell interfaces (nz+1, ny, nx)
                 defined at top of cell k.
                 kappa_z[k] is diff at interface k-1/2 (between k-1 and k).
                 kappa_z[0] = 0 (surface), kappa_z[nz] = 0 (bottom).
    """
    nz, ny, nx = rho.shape
    
    # Calculate stability N^2 ~ drho/dz
    # We need drho/dz at interfaces.
    # rho[k] is cell center. Interface k is betwen k-1 and k.
    
    # dz between centers
    dz_3d = dz.reshape(-1, 1, 1)
    dist = 0.5 * (dz_3d[:-1] + dz_3d[1:])
    
    # drho at interfaces 1..nz-1
    drho = rho[1:] - rho[:-1]
    
    # Check stability.
    # Standard: rho increases with depth (index increases).
    # Stable: rho[k] < rho[k+1] => drho > 0.
    # Unstable: drho < 0.
    
    # Define kappa at interfaces
    # Interior interfaces 1..nz-1
    is_unstable = drho < 0
    kappa_interior = jnp.where(is_unstable, kappa_convect, kappa_bg)
    
    # Pad top and bottom (zero flux BC implies 0 diffusivity effectively, 
    # but strictly diffusivity can be non-zero if flux is applied.
    # usually kappa at boundary is used for surface flux penetration if not applied as source.
    # Here we treat surface flux as source term in T equation, so kappa_z[0]=0 is consistent for diffusion operator.
    
    zeros = jnp.zeros((1, ny, nx))
    kappa_z = jnp.concatenate([zeros, kappa_interior, zeros], axis=0)
    
    return kappa_z
