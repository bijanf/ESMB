"""
JAX-native Conjugate Gradient solver for ocean streamfunction equation.

This module provides a fully differentiable CG solver that can be JIT-compiled
and used in gradient-based optimization. It replaces scipy.sparse.linalg.cg
which is not differentiable.

The solver is designed to solve the barotropic streamfunction equation:
    ∇ · (H^{-1} ∇ψ) = ζ
where ψ is the streamfunction and ζ is the vorticity.
"""

from functools import partial
from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp


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


@partial(jax.jit, static_argnames=["max_iter", "linear_operator", "preconditioner"])
def solve_cg(
    linear_operator: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    x0: jnp.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
    preconditioner: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
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

    # Use jax.lax.scan with fixed iterations (differentiable in reverse mode).
    # while_loop is NOT differentiable. scan runs exactly max_iter steps
    # but uses jnp.where to stop updating once converged (no-op iterations).
    def scan_body(state, _):
        x, r, p, rz_old, r_norm, iter_count = state
        # Only update if not yet converged
        converged = r_norm <= tol_abs
        new_state = body_fun(state)
        # If converged, keep old state (no-op)
        out_state = jax.tree.map(
            lambda old, new: jnp.where(converged, old, new), state, new_state)
        return out_state, None

    initial_state_loop = (x0, r0, p0, rz0, jnp.linalg.norm(r0), 0)
    final_state, _ = jax.lax.scan(scan_body, initial_state_loop,
                                   None, length=max_iter)

    x_final, r_final, _, _, r_norm_final, num_iters = final_state

    # We don't have full history in while_loop easily without creating a big array
    # Just return final residual
    residual_history = jnp.array([r_norm_final])  # Dummy history

    converged = r_norm_final < tol_abs
    min_residual = r_norm_final

    info = {
        "num_iters": num_iters,
        "residual_norm": min_residual,
        "converged": converged,
        "residual_history": residual_history,
    }

    return x_final, info


def create_diagonal_preconditioner(
    linear_operator: Callable[[jnp.ndarray], jnp.ndarray], n: int
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


@partial(jax.jit, static_argnames=["max_iter"])
def solve_poisson_2d(
    rhs: jnp.ndarray,
    dx: float,
    dy: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
    mask: Optional[jnp.ndarray] = None,  # 1=Ocean, 0=Land
    x0: Optional[jnp.ndarray] = None,
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
    b = -rhs.flatten()  # Flip sign of RHS

    if x0 is not None:
        x0_flat = x0.flatten()
    else:
        x0_flat = jnp.zeros_like(b)

    # Define Laplacian operator with periodic BC
    def laplacian_operator(x_flat):
        x_2d = x_flat.reshape((ny, nx))

        # Second derivatives with periodic BC (using roll)
        d2x = (
            jnp.roll(x_2d, 1, axis=1) - 2 * x_2d + jnp.roll(x_2d, -1, axis=1)
        ) / dx**2
        # Y-axis: Dirichlet BC (0 at boundaries). Pad with 0.
        x_padded_y = jnp.pad(x_2d, ((1, 1), (0, 0)), mode="constant", constant_values=0)
        d2y = (x_padded_y[:-2, :] - 2 * x_2d + x_padded_y[2:, :]) / dy**2

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

        return -laplacian.flatten()  # Flip sign of operator (Positive Definite)

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
        preconditioner=precond,
    )

    return x_flat.reshape((ny, nx)), info


# ---------------------------------------------------------------------------
# Variable-coefficient elliptic solver (P3 / S2): div(coef * grad psi) = rhs.
# The load-bearing invert for a prognostic barotropic-vorticity / JEBAR ocean core.
# coef = 1/H (cell-centred) gives the topographic/JEBAR operator over real bathymetry;
# coef = const reproduces the Poisson solve. Periodic in x, Dirichlet psi=0 in y and on
# land (the identity-row trick from solve_poisson_2d). Wraps the AD-safe scan-CG, so it
# is differentiable. Face coefficients are the arithmetic mean of adjacent cells, zeroed
# on closed (land) faces so no flux crosses coastlines.
# ---------------------------------------------------------------------------
def _varcoef_faces(coef):
    """East/West/North/South face coefficients = arithmetic mean of adjacent cells.

    Faces are NOT masked: ocean cells couple to neighbouring land cells, which are pinned
    to psi=0 by the identity rows in the operator -> a Dirichlet (psi=0) coast/boundary,
    which is the correct streamfunction BC. (Masking the faces would impose a no-flux /
    Neumann condition -> a singular operator and a divergent CG.) coef must be finite
    everywhere (regularise 1/H on land before calling)."""
    cE = 0.5 * (coef + jnp.roll(coef, -1, axis=1))
    cW = 0.5 * (coef + jnp.roll(coef, 1, axis=1))
    coef_n = jnp.concatenate([coef[1:, :], coef[-1:, :]], axis=0)
    coef_s = jnp.concatenate([coef[:1, :], coef[:-1, :]], axis=0)
    cN = 0.5 * (coef + coef_n)
    cS = 0.5 * (coef + coef_s)
    return cE, cW, cN, cS


def _elliptic_op(psi, cE, cW, cN, cS, dx2, dy2):
    """Discrete div(coef * grad psi) on the periodic-x / Dirichlet-y grid."""
    pe = jnp.roll(psi, -1, axis=1)
    pw = jnp.roll(psi, 1, axis=1)
    pn = jnp.concatenate([psi[1:, :], jnp.zeros_like(psi[:1, :])], axis=0)
    ps = jnp.concatenate([jnp.zeros_like(psi[:1, :]), psi[:-1, :]], axis=0)
    flux_x = (cE * (pe - psi) - cW * (psi - pw)) / dx2
    flux_y = (cN * (pn - psi) - cS * (psi - ps)) / dy2
    return flux_x + flux_y


def apply_elliptic_varcoef(psi, coef, dx, dy, mask=None):
    """Apply L psi = div(coef * grad psi) (ocean cells only). Useful for manufactured
    solutions and for a Laplacian-of-vorticity viscous term."""
    ny, nx = psi.shape
    coef = jnp.broadcast_to(jnp.asarray(coef, psi.dtype), (ny, nx))
    mask = jnp.ones((ny, nx), psi.dtype) if mask is None else mask.astype(psi.dtype)
    cE, cW, cN, cS = _varcoef_faces(coef)
    return _elliptic_op(psi, cE, cW, cN, cS,
                        jnp.asarray(dx) ** 2, jnp.asarray(dy) ** 2) * mask


def solve_elliptic_varcoef(coef, rhs, dx, dy, mask=None, x0=None,
                           max_iter=400, tol=1e-7):
    """Solve div(coef * grad psi) = rhs for psi. Returns (psi, info).

    coef, rhs: (ny, nx). dx, dy: scalar or broadcastable. mask: 1=ocean, 0=land
    (psi=0 enforced on land). x0: warm-start. Differentiable in coef, rhs (and x0).
    """
    ny, nx = rhs.shape
    coef = jnp.broadcast_to(jnp.asarray(coef, rhs.dtype), (ny, nx))
    mask = jnp.ones((ny, nx), rhs.dtype) if mask is None else mask.astype(rhs.dtype)
    cE, cW, cN, cS = _varcoef_faces(coef)
    dx2, dy2 = jnp.asarray(dx) ** 2, jnp.asarray(dy) ** 2

    def operator(psi_flat):
        psi = psi_flat.reshape(ny, nx)
        lap = _elliptic_op(psi, cE, cW, cN, cS, dx2, dy2)
        # A = -L on ocean (SPD), identity on land -> psi=0 where rhs=0
        return (-lap * mask + psi * (1.0 - mask)).flatten()

    diag = (cE + cW) / dx2 + (cN + cS) / dy2          # diag of -L (positive)
    diag = diag * mask + (1.0 - mask)                 # identity on land
    precond = jacobi_preconditioner(diag.flatten())

    b = (-rhs * mask).flatten()
    x0_flat = x0.flatten() if x0 is not None else jnp.zeros_like(b)
    x, info = solve_cg(operator, b, x0_flat, max_iter=max_iter, tol=tol,
                       preconditioner=precond)
    return x.reshape(ny, nx), info


# ---------------------------------------------------------------------------
# SPHERICAL variable-coefficient elliptic solver (P3 / S2, second half):
#   (1/(a^2 cos(phi))) [ d/dlon (coef/cos(phi) dpsi/dlon)
#                      + d/dphi (coef cos(phi) dpsi/dphi) ] = rhs
# This is the operator a prognostic barotropic-vorticity / JEBAR ocean core needs on the
# real lat-lon grid (coef = 1/H over ETOPO bathymetry). It is discretised in symmetric
# FACE-CONDUCTANCE form (multiply the cell equation by the cell area a^2 cos(phi) dlon
# dlat) so the matrix stays symmetric positive-definite and the AD-safe CG applies -- the
# area-divided form used in the Cartesian solver would be non-symmetric once the cell area
# varies with latitude. A polar cos(phi) floor avoids the 1/cos(phi) blow-up at the lat-lon
# pole singularity (same idea as the atmosphere's pole-dx floor). Periodic in lon, Dirichlet
# psi=0 in lat and on land. Differentiable in coef, rhs.
# ---------------------------------------------------------------------------
def _sphere_conductances(coef, lat, dlon, dlat, a, cos_min):
    """Symmetric east/west/north/south face conductances + cell area on the sphere.

    coef: (ny, nx) cell-centred (e.g. 1/H). lat: (ny,) cell-centre latitude [rad].
    dlon, dlat: grid spacing [rad]. a: Earth radius [m]. cos_min: polar cos floor.
    """
    cosc = jnp.maximum(jnp.cos(lat), cos_min)[:, None]                 # (ny,1) cell centre
    lat_n = 0.5 * (lat + jnp.concatenate([lat[1:], lat[-1:]]))         # north-face latitude
    lat_s = 0.5 * (lat + jnp.concatenate([lat[:1], lat[:-1]]))         # south-face latitude
    cosn = jnp.maximum(jnp.cos(lat_n), cos_min)[:, None]
    coss = jnp.maximum(jnp.cos(lat_s), cos_min)[:, None]
    # face coef = arithmetic mean of adjacent cells (Dirichlet via land-identity, not masked)
    coef_n = jnp.concatenate([coef[1:, :], coef[-1:, :]], axis=0)
    coef_s = jnp.concatenate([coef[:1, :], coef[:-1, :]], axis=0)
    # conductance = coef_face * face_length / centre_distance (the Earth radius cancels):
    #   E/W face length = a*dlat, distance = a*cos(phi)*dlon -> dlat/(cos*dlon)
    #   N/S face length = a*cos(phi_face)*dlon, distance = a*dlat -> cos_face*dlon/dlat
    cE = 0.5 * (coef + jnp.roll(coef, -1, axis=1)) * (dlat / (cosc * dlon))
    cW = 0.5 * (coef + jnp.roll(coef, 1, axis=1)) * (dlat / (cosc * dlon))
    cN = 0.5 * (coef + coef_n) * (cosn * dlon / dlat)
    cS = 0.5 * (coef + coef_s) * (coss * dlon / dlat)
    area = (a ** 2) * cosc * dlon * dlat                               # (ny,1) cell area
    return cE, cW, cN, cS, area


def _sphere_op(psi, cE, cW, cN, cS):
    """M psi = sum_faces C_face (psi - psi_nbr) = -area * div(coef grad psi) (symmetric)."""
    pe = jnp.roll(psi, -1, axis=1)
    pw = jnp.roll(psi, 1, axis=1)
    pn = jnp.concatenate([psi[1:, :], jnp.zeros_like(psi[:1, :])], axis=0)
    ps = jnp.concatenate([jnp.zeros_like(psi[:1, :]), psi[:-1, :]], axis=0)
    return (cE * (psi - pe) + cW * (psi - pw)
            + cN * (psi - pn) + cS * (psi - ps))


def apply_elliptic_varcoef_sphere(psi, coef, lat, dlon, dlat, a, mask=None,
                                  cos_min=0.087):
    """Apply the spherical operator L psi = div(coef grad psi) (area-divided, ocean only)."""
    ny, nx = psi.shape
    coef = jnp.broadcast_to(jnp.asarray(coef, psi.dtype), (ny, nx))
    mask = jnp.ones((ny, nx), psi.dtype) if mask is None else mask.astype(psi.dtype)
    cE, cW, cN, cS, area = _sphere_conductances(coef, jnp.asarray(lat, psi.dtype),
                                                dlon, dlat, a, cos_min)
    return (-_sphere_op(psi, cE, cW, cN, cS) / area) * mask


def solve_elliptic_varcoef_sphere(coef, rhs, lat, dlon, dlat, a, mask=None, x0=None,
                                  max_iter=600, tol=1e-7, cos_min=0.087):
    """Solve the spherical div(coef grad psi) = rhs for psi. Returns (psi, info).

    coef, rhs: (ny, nx). lat: (ny,) cell-centre latitude [rad]. dlon, dlat: [rad].
    a: Earth radius [m]. mask: 1=ocean,0=land (psi=0 on land). Differentiable.
    """
    ny, nx = rhs.shape
    coef = jnp.broadcast_to(jnp.asarray(coef, rhs.dtype), (ny, nx))
    mask = jnp.ones((ny, nx), rhs.dtype) if mask is None else mask.astype(rhs.dtype)
    lat = jnp.asarray(lat, rhs.dtype)
    cE, cW, cN, cS, area = _sphere_conductances(coef, lat, dlon, dlat, a, cos_min)

    def operator(psi_flat):
        psi = psi_flat.reshape(ny, nx)
        m = _sphere_op(psi, cE, cW, cN, cS)               # symmetric SPD on ocean
        return (m * mask + psi * (1.0 - mask)).flatten()

    diag = (cE + cW + cN + cS) * mask + (1.0 - mask)      # diag of M (positive), id on land
    precond = jacobi_preconditioner(diag.flatten())

    b = (-rhs * area * mask).flatten()                    # symmetric: multiply by cell area
    x0_flat = x0.flatten() if x0 is not None else jnp.zeros_like(b)
    x, info = solve_cg(operator, b, x0_flat, max_iter=max_iter, tol=tol,
                       preconditioner=precond)
    return x.reshape(ny, nx), info
