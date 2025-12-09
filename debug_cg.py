"""Test basic CG with step-by-step trace."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp

# Simple 3x3 diagonal  system
A = jnp.array([[2.0, 0.0, 0.0],
               [0.0, 3.0, 0.0],
               [0.0, 0.0, 4.0]])
b = jnp.array([2.0, 6.0, 12.0])
x_true = jnp.array([1.0, 2.0, 3.0])

# Verify true solution
print("Verification:")
print(f"A @ x_true = {A @ x_true}")
print(f"b = {b}")
print(f"Match: {jnp.allclose(A @ x_true, b)}")

# CG should solve this in 3 iterations for a 3x3 diagonal
# Let's use a more reasonable tolerance
from chronos_esm.ocean.solver import solve_cg

A_func = lambda x: A @ x
x0 = jnp.zeros(3)

print("\n=== Test with tol=1e-8 ===")
x_cg, info = solve_cg(A_func, b, x0, max_iter=10, tol=1e-8)
print(f"Solution: {x_cg}")
print(f"True: {x_true}")
print(f"Error: {jnp.linalg.norm(x_cg - x_true)}")
print(f"Converged: {info['converged']}")
print(f"Iterations: {info['num_iters']}")
print(f"Residual: {info['residual_norm']:.2e}")

print("\n=== Test with tol=1e-5 ===")
x_cg2, info2 = solve_cg(A_func, b, x0, max_iter=10, tol=1e-5)
print(f"Converged: {info2['converged']}")
print(f"Iterations: {info2['num_iters']}")
print(f"Residual: {info2['residual_norm']:.2e}")
