"""Proxy / differentiable-data-assimilation layer for Chronos-ESM.

This package is a differentiable LIBRARY of the pieces needed for gradient-based
proxy fitting and data assimilation:

* ``forward_ops`` -- state -> proxy operators (the Bemis 1998 d18O-calcite operator
  and the LeGrande & Schmidt 2006 salinity -> d18O_sw relation), all jax-grad-safe;
* ``sensor``      -- sparse-observation sampling (nearest + bilinear) and masking;
* ``loss``        -- masked MSE / climatology losses.

SCOPE (honest): these are validated, differentiable building blocks. An end-to-end
4D-Var assimilation loop that wires the forward operators into the coupled model and
optimises against real proxy observations is FUTURE WORK -- the gradient pathway
itself is already demonstrated (see tests/test_adjoint.py and experiments/
verify_gradient.py). See docs/manual ch15 (Differentiable Applications) and
appendix E (Model Limitations).
"""
