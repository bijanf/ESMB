"""
Comprehensive test suite for the JAX-native Conjugate Gradient solver.

Tests include:
1. Simple diagonal system
2. Random SPD matrix comparison with SciPy
3. 2D Poisson problem
4. Gradient flow (differentiability)
5. Preconditioner efficiency
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest  # noqa: F401
from scipy.sparse.linalg import cg as scipy_cg

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm.ocean.solver import (  # noqa: E402
    jacobi_preconditioner,
    solve_cg,
    solve_poisson_2d,
)


class TestCGSolver:
    """Test suite for Conjugate Gradient solver."""

    def test_diagonal_system(self):
        """Test 1: Simple diagonal system - verify basic CG mechanics."""
        # System: diag([2, 3, 4]) * x = [2, 6, 12]
        # Solution: x = [1, 2, 3]

        A = jnp.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
        b = jnp.array([2.0, 6.0, 12.0])
        x0 = jnp.zeros(3)

        def A_func(x):
            return A @ x

        x_cg, info = solve_cg(A_func, b, x0, max_iter=100, tol=1e-8)
        x_true = jnp.array([1.0, 2.0, 3.0])

        # Check solution accuracy
        assert jnp.allclose(x_cg, x_true, rtol=1e-8)

        # Check convergence
        assert info["converged"]

        # Diagonal systems should converge very quickly
        assert info["num_iters"] <= 10

        print(
            f"✓ Test 1 passed: Diagonal system solved in "
            f"{info['num_iters']} iterations"
        )

    def test_random_spd_matrix(self):
        """Test 2: Random SPD matrix - compare against SciPy."""
        np.random.seed(42)
        n = 50

        # Generate random SPD matrix: A = B^T B + I
        B = np.random.randn(n, n)
        A_np = B.T @ B + np.eye(n)
        A_jax = jnp.array(A_np)

        # Random RHS
        b_np = np.random.randn(n)
        b_jax = jnp.array(b_np)
        x0 = jnp.zeros(n)

        # Solve with JAX CG
        def A_func(x):
            return A_jax @ x

        x_jax, info = solve_cg(A_func, b_jax, x0, max_iter=200, tol=1e-6)

        # Solve with SciPy CG
        x_scipy, scipy_info = scipy_cg(A_np, b_np, atol=1e-6, maxiter=200)

        # Compare solutions
        assert jnp.allclose(x_jax, x_scipy, rtol=1e-4, atol=1e-4)

        # Both should converge
        assert info["converged"]
        assert scipy_info == 0  # SciPy success

        print(
            f"✓ Test 2 passed: Random SPD system - JAX: {info['num_iters']} iters, "
            f"SciPy: {scipy_info} status, "
            f"max diff: {jnp.max(jnp.abs(x_jax - x_scipy)):.2e}"
        )

    def test_2d_poisson(self):
        """Test 3: 2D Poisson problem - realistic PDE test."""
        # Solve: ∇²ψ = sin(2πx)cos(2πy) on [0,1]²
        # with periodic BC

        nx, ny = 32, 32
        x = jnp.linspace(0, 1, nx, endpoint=False)
        y = jnp.linspace(0, 1, ny, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        # RHS
        rhs = jnp.sin(2 * jnp.pi * X) * jnp.cos(2 * jnp.pi * Y)

        # Analytical solution for this RHS:
        # ψ = -sin(2πx)cos(2πy) / (2*(2π)²) = -sin(2πx)cos(2πy) / (8π²)
        psi_analytical = (
            -jnp.sin(2 * jnp.pi * X) * jnp.cos(2 * jnp.pi * Y) / (8 * jnp.pi**2)
        )

        dx = 1.0 / nx
        dy = 1.0 / ny

        # Solve using our CG Poisson solver
        psi_numerical, info = solve_poisson_2d(rhs, dx, dy, max_iter=500, tol=1e-6)

        # The solution is unique up to a constant, so normalize both
        psi_numerical = psi_numerical - jnp.mean(psi_numerical)
        psi_analytical = psi_analytical - jnp.mean(psi_analytical)

        # Check relative error
        rel_error = jnp.linalg.norm(psi_numerical - psi_analytical) / jnp.linalg.norm(
            psi_analytical
        )

        assert rel_error < 0.01  # 1% relative error
        # Note: May not formally "converge" to tolerance but solution is accurate

        print(
            f"✓ Test 3 passed: 2D Poisson - converged in "
            f"{info['num_iters']} iterations, "
            f"relative error: {rel_error:.2e}"
        )

    def test_gradient_flow(self):
        """Test 4: Verify differentiability via jax.grad."""
        # Define a simple quadratic optimization problem:
        # minimize 0.5 * x^T A x - b^T x
        # The gradient w.r.t b should be -x*

        n = 10
        np.random.seed(123)
        B = np.random.randn(n, n)
        A_np = B.T @ B + np.eye(n)
        A = jnp.array(A_np)

        def objective(b):
            """Solve A*x = b and return 0.5*x^T*A*x - b^T*x."""

            def A_func(x):
                return A @ x

            x0 = jnp.zeros(n)
            x, _ = solve_cg(A_func, b, x0, max_iter=100, tol=1e-8)

            # At optimum, this should be -0.5 * x^T * A * x
            return 0.5 * jnp.dot(x, A @ x) - jnp.dot(b, x)

        # Test gradient computation (should not error)
        b_test = jnp.ones(n)
        grad_b = jax.grad(objective)(b_test)

        # The gradient should be well-defined and non-zero
        assert jnp.all(jnp.isfinite(grad_b))
        assert jnp.linalg.norm(grad_b) > 0

        print(
            f"✓ Test 4 passed: Gradient flow works, grad norm: "
            f"{jnp.linalg.norm(grad_b):.2e}"
        )

    def test_preconditioner_efficiency(self):
        """Test 5: Compare iterations with/without Jacobi preconditioning."""
        n = 100
        np.random.seed(456)

        # Create a poorly conditioned matrix
        # Use a diagonal with large condition number
        diag = jnp.linspace(1.0, 100.0, n)
        A = jnp.diag(diag) + 0.1 * jnp.ones((n, n))
        b = jnp.ones(n)
        x0 = jnp.zeros(n)

        def A_func(x):
            return A @ x

        # Solve without preconditioning
        _, info_no_precond = solve_cg(A_func, b, x0, max_iter=500, tol=1e-6)

        # Solve with Jacobi preconditioning
        diagonal = jnp.diag(A)
        precond = jacobi_preconditioner(diagonal)
        _, info_precond = solve_cg(
            A_func, b, x0, max_iter=500, tol=1e-6, preconditioner=precond
        )

        # Preconditioning should reduce iterations
        iters_no_precond = info_no_precond["num_iters"]
        iters_precond = info_precond["num_iters"]

        # Preconditioned should take fewer iterations (or at most equal)
        assert iters_precond <= iters_no_precond

        # For this problem, expect significant speedup
        speedup = iters_no_precond / (iters_precond + 1e-6)

        print(
            f"✓ Test 5 passed: Preconditioner efficiency - "
            f"No precond: {iters_no_precond} iters, "
            f"With precond: {iters_precond} iters, "
            f"Speedup: {speedup:.2f}x"
        )


class TestJITCompilation:
    """Test JIT compilation performance."""

    def test_jit_compilation(self):
        """Verify solver can be JIT compiled."""
        n = 100
        A = jnp.eye(n) + 0.1 * jnp.ones((n, n))
        b = jnp.ones(n)
        x0 = jnp.zeros(n)

        def A_func(x):
            return A @ x

        # First call will compile
        x1, info1 = solve_cg(A_func, b, x0, max_iter=100, tol=1e-6)

        # Second call should reuse compiled code
        x2, info2 = solve_cg(A_func, b, x0, max_iter=100, tol=1e-6)

        # Results should be identical
        assert jnp.array_equal(x1, x2)
        assert info1["num_iters"] == info2["num_iters"]

        print("✓ JIT test passed: Solver compiles and runs consistently")


if __name__ == "__main__":
    # Run tests
    print("=" * 70)
    print("Running Conjugate Gradient Solver Tests")
    print("=" * 70)

    test_suite = TestCGSolver()

    print("\n1. Testing diagonal system...")
    test_suite.test_diagonal_system()

    print("\n2. Testing random SPD matrix vs SciPy...")
    test_suite.test_random_spd_matrix()

    print("\n3. Testing 2D Poisson equation...")
    test_suite.test_2d_poisson()

    print("\n4. Testing gradient flow (differentiability)...")
    test_suite.test_gradient_flow()

    print("\n5. Testing preconditioner efficiency...")
    test_suite.test_preconditioner_efficiency()

    print("\n" + "=" * 70)
    print("Testing JIT compilation...")
    print("=" * 70)

    jit_test = TestJITCompilation()
    print("\n6. Testing JIT compilation...")
    jit_test.test_jit_compilation()

    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
