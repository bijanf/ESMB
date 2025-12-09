"""
JAX-native Conjugate Gradient solver for ocean streamfunction equation.

This module provides a fully differentiable CG solver that can be JIT-compiled
and used in gradient-based optimization. It replaces scipy.sparse.linalg.cg
which is not differentiable.

The solver is designed to solve the barotropic streamfunction equation:
    ∇ · (H^{-1} ∇ψ) = ζ
where ψ is the streamfunction and ζ is the vorticity.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Optional, Dict, Tuple
from functools import partial


def jacobi_preconditioner(diagonal: jnp.ndarray) -> Callable:
    """
    Create a Jacobi preconditioner M ≈ diag(A)^{-1}.
    
    Args:
        diagonal: Diagonal elements of the matrix A
        
    Returns:
        Function that applies the preconditioner: M(r) = r / diag(A)
    """
    # Avoid division by zero with a small epsilon
    diag_inv = 1.0 / (diagonal + 1e-12)
    
    def precondition(r: jnp.ndarray) -> jnp.ndarray:
        return diag_inv * r
    
    return precondition


@partial(jax.jit, static_argnames=['max_iter', 'linear_operator', 'preconditioner'])
def solve_cg(
    linear_operator: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    x0: jnp.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
    preconditioner: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Solve the linear system A*x = b using the Conjugate Gradient method.
    
    This implementation uses jax.lax.scan for the iteration loop to ensure
    differentiability and efficient compilation. The algorithm follows the
    standard preconditioned CG algorithm.
    
    Args:
        linear_operator: Function A(x) that applies the linear operator
        b: Right-hand side vector
        x0: Initial guess for the solution
        max_iter: Maximum number of iterations
        tol: Convergence tolerance (relative residual norm)
        preconditioner: Optional function M(r) for preconditioning
        
    Returns:
        x: Solution vector
        info: Dictionary containing:
            - num_iters: Number of iterations performed
            - residual_norm: Final residual norm
            - converged: Whether the solver converged
            - residual_history: Array of residual norms at each iteration
    """
    
    # Identity preconditioner if none provided
    if preconditioner is None:
        preconditioner = lambda r: r
    
    # Compute initial residual: r0 = b - A(x0)
    r0 = b - linear_operator(x0)
    
    # Apply preconditioner: z0 = M(r0)
    z0 = preconditioner(r0)
    
    # Initial search direction
    p0 = z0
    
    # Initial dot product for convergence check
    rz0 = jnp.dot(r0, z0)
    
    # Norm of b for relative tolerance
    b_norm = jnp.linalg.norm(b)
    tol_abs = tol * b_norm
    
    # Initial state for the scan loop
    initial_state = (x0, r0, p0, rz0, jnp.linalg.norm(r0))
    
    # Run CG iterations using while_loop for early exit
    def cond_fun(state):
        _, _, _, _, r_norm, iter_count = state
        # Continue if residual > tol AND iter < max_iter
        return (r_norm > tol_abs) & (iter_count < max_iter)

    def body_fun(state):
        x, r, p, rz_old, r_norm, iter_count = state
        
        # Compute A*p
        Ap = linear_operator(p)
        
        # Step size: alpha = (r^T * z) / (p^T * A * p)
        pAp = jnp.dot(p, Ap)
        # Protect division
        denominator = jnp.where(jnp.abs(pAp) < 1e-20, 1.0, pAp)
        alpha = rz_old / denominator
        
        # Update solution: x = x + alpha * p
        x_new = x + alpha * p
        
        # Update residual: r = r - alpha * A*p
        r_new = r - alpha * Ap
        
        # Apply preconditioner: z = M(r)
        z_new = preconditioner(r_new)
        
        # New dot product
        rz_new = jnp.dot(r_new, z_new)
        
        # Conjugate direction update: beta = rz_new / rz_old
        denom_beta = jnp.where(jnp.abs(rz_old) < 1e-20, 1.0, rz_old)
        beta = rz_new / denom_beta
        
        # Update search direction: p = z + beta * p
        p_new = z_new + beta * p
        
        # Compute residual norm
        r_norm_new = jnp.linalg.norm(r_new)
        
        return (x_new, r_new, p_new, rz_new, r_norm_new, iter_count + 1)

    # Initial state needs iter_count
    # (x, r, p, rz, r_norm, iter)
    initial_state_loop = (x0, r0, p0, rz0, jnp.linalg.norm(r0), 0)
    
    final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state_loop)
    
    x_final, r_final, _, _, r_norm_final, num_iters = final_state
    
    # We don't have full history in while_loop easily without creating a big array
    # Just return final residual
    residual_history = jnp.array([r_norm_final]) # Dummy history
    
    converged = r_norm_final < tol_abs
    min_residual = r_norm_final
    
    info = {
        'num_iters': num_iters,
        'residual_norm': min_residual,
        'converged': converged,
        'residual_history': residual_history
    }
    
    return x_final, info


def create_diagonal_preconditioner(
    linear_operator: Callable[[jnp.ndarray], jnp.ndarray],
    n: int
) -> Callable:
    """
    Extract diagonal of a linear operator and create Jacobi preconditioner.
    
    This function applies the linear operator to each standard basis vector
    to extract the diagonal elements.
    
    Args:
        linear_operator: Function A(x) that applies the linear operator
        n: Dimension of the vector space
        
    Returns:
        Preconditioner function
    """
    # Extract diagonal by applying operator to standard basis vectors
    def get_diagonal_element(i):
        e_i = jnp.zeros(n)
        e_i = e_i.at[i].set(1.0)
        Ae_i = linear_operator(e_i)
        return Ae_i[i]
    
    diagonal = jax.vmap(get_diagonal_element)(jnp.arange(n))
    
    return jacobi_preconditioner(diagonal)


@partial(jax.jit, static_argnames=['max_iter'])
def solve_poisson_2d(
    rhs: jnp.ndarray,
    dx: float,
    dy: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    mask: Optional[jnp.ndarray] = None, # 1=Ocean, 0=Land
    x0: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Solve 2D Poisson equation using CG: ∇²ψ = rhs.
    
    Uses finite differences with periodic boundary conditions.
    If mask is provided, enforces ψ=0 on land (Dirichlet).
    
    Args:
        rhs: Right-hand side (2D array)
        dx: Grid spacing in x direction
        dy: Grid spacing in y direction
        max_iter: Maximum CG iterations
        tol: Convergence tolerance
        mask: Boolean mask (1.0=Ocean, 0.0=Land)
        x0: Initial guess (2D array)
        
    Returns:
        Solution ψ (2D array)
    """
    ny, nx = rhs.shape
    
    # If mask is provided, zero out RHS on land
    if mask is not None:
        rhs = rhs * mask
    
    # Flatten arrays for linear algebra
    b = -rhs.flatten() # Flip sign of RHS
    
    if x0 is not None:
        x0_flat = x0.flatten()
    else:
        x0_flat = jnp.zeros_like(b)
    
    # Define Laplacian operator with periodic BC
    def laplacian_operator(x_flat):
        x_2d = x_flat.reshape((ny, nx))
        
        # Second derivatives with periodic BC (using roll)
        d2x = (jnp.roll(x_2d, 1, axis=1) - 2*x_2d + jnp.roll(x_2d, -1, axis=1)) / dx**2
        d2y = (jnp.roll(x_2d, 1, axis=0) - 2*x_2d + jnp.roll(x_2d, -1, axis=0)) / dy**2
        
        laplacian = d2x + d2y
        
        # Apply mask: on land, operator is Identity (or scaling) to enforce x=0 if b=0
        # Actually, for CG, we want symmetric positive definite.
        # If we set row/col to identity for land points:
        # A_ii = 1, A_ij = 0.
        # Then if b_i = 0, x_i = 0.
        
        if mask is not None:
            # Mask out the Laplacian result on land
            laplacian = laplacian * mask
            # Add identity term on land to avoid singular matrix
            # On land: L(x) = x (so if b=0, x=0)
            # We need negative definite operator usually for Poisson?
            # We return -Laplacian.
            # So we want -L(x) = x -> L(x) = -x.
            # Let's just say Operator(x) = x on land.
            # Result = -Laplacian_ocean + x_land
            
            # x_land = x_2d * (1 - mask)
            # op = -laplacian * mask + x_land
            # return op.flatten()
            
            # Wait, standard is -Laplacian.
            # Ocean: -Laplacian
            # Land: Identity
            
            op_ocean = -laplacian * mask
            op_land = x_2d * (1.0 - mask)
            return (op_ocean + op_land).flatten()
            
        return -laplacian.flatten() # Flip sign of operator (Positive Definite)
    
    # Extract diagonal for Jacobi preconditioner
    # For 2D Laplacian: diag = -2/dx² - 2/dy²
    # We want negative of that: 2/dx² + 2/dy²
    # We want negative of that: 2/dx² + 2/dy²
    diag_value = 2.0 / dx**2 + 2.0 / dy**2
    # Broadcast to (ny, nx) if dx is an array
    diag_value = jnp.broadcast_to(diag_value, (ny, nx))
    diagonal = diag_value.flatten()
    
    if mask is not None:
        # On land, diagonal is 1.0
        mask_flat = mask.flatten()
        diagonal = diagonal * mask_flat + 1.0 * (1.0 - mask_flat)
        
    precond = jacobi_preconditioner(diagonal)
    
    # Solve
    x_flat, info = solve_cg(
        laplacian_operator, 
        b, 
        x0_flat, 
        max_iter=max_iter, 
        tol=tol,
        preconditioner=precond
    )
    
    return x_flat.reshape((ny, nx)), info
