# Chronos-ESM

A fully differentiable, coupled Earth System Model (T63 Resolution) implemented in JAX.

## Overview

Chronos-ESM is a cutting-edge Earth System Model designed for:
- **End-to-end differentiability** for proxy data assimilation
- **Memory bandwidth optimization** via XLA fusion
- **Consumer hardware** (14-core CPU, 32GB RAM)

The system couples:
- **Ocean**: Veros primitive equations at T63 (~192×96×40)
- **Atmosphere**: QTCM (Quasi-Equilibrium Tropical Circulation Model) with spectral transforms
- **Proxy Layer**: Bemis isotope equations for paleo-data assimilation

## Project Structure

```
chronos_esm/
├── config.py             # Global T63 grid definitions & constants
├── ocean/
│   ├── solver.py         # JAX-native Conjugate Gradient solver ✅
│   ├── veros_driver.py   # JAX wrapper for Veros physics
│   └── mixing.py         # Differentiable GM & Isopycnal schemes
├── atmos/
│   ├── qtcm.py           # Main atmospheric stepper
│   ├── spectral.py       # Spherical Harmonic Transforms
│   └── physics.py        # Differentiable moist convection
├── coupler/
│   ├── regrid.py         # Dense matrix multiplication for conservation
│   └── state.py          # Dataclasses for coupled state
├── proxy/
│   ├── forward_ops.py    # Bemis equations & Sensor masking
│   └── loss.py           # Time-averaged climatology loss
└── main.py               # The training loop
```

## Installation

### Requirements
- Python 3.9+
- JAX (CPU or GPU version)

### Setup

```bash
# Clone the repository
cd /home/bijanf/Documents/ESMB

# Install dependencies
pip install -r requirements.txt

# Verify JAX installation
python -c "import jax; print(jax.devices())"
```

## Quick Start

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run solver tests only
pytest tests/test_solver.py -v

# Or run directly
python tests/test_solver.py
```

### Example: Using the CG Solver

```python
from chronos_esm.ocean.solver import solve_cg
import jax.numpy as jnp

# Define a linear system: A*x = b
A = jnp.array([[4., 1., 0.], 
               [1., 4., 1.], 
               [0., 1., 4.]])
b = jnp.array([1., 2., 3.])

# Define linear operator
A_func = lambda x: A @ x

# Solve
x0 = jnp.zeros(3)
x_solution, info = solve_cg(A_func, b, x0, max_iter=100, tol=1e-8)

print(f"Solution: {x_solution}")
print(f"Converged in {info['num_iters']} iterations")
print(f"Final residual: {info['residual_norm']:.2e}")
```

## Current Status

✅ **Phase 1 Complete**: Ocean CG Solver
- JAX-native Conjugate Gradient implementation
- Jacobi preconditioning
- Full differentiability via `jax.grad`
- Comprehensive test suite

🚧 **In Progress**: Additional components coming soon

## Hardware Target

- **CPU**: 14 cores
- **RAM**: 32GB
- **Optimization**: XLA fusion for memory bandwidth

## Key Features

### Differentiability
All components are implemented using JAX primitives to ensure end-to-end differentiability for gradient-based optimization and data assimilation.

### Memory Efficiency
Uses `jax.checkpoint` and `jax.lax.scan` to manage memory during long time integrations (10+ years).

### Scientific Accuracy
Implements established numerical methods:
- Conjugate Gradient for elliptic PDEs
- Spectral transforms for atmospheric dynamics
- Conservative regridding for coupling

## Contributing

This is a research project. For questions or collaboration, please open an issue.

## License

MIT License (or specify your license)

## References

- Veros ocean model: https://veros.readthedocs.io/
- QTCM atmospheric model: Neelin & Zeng (2000)
- JAX documentation: https://jax.readthedocs.io/
