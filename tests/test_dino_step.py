"""P1 tests: the multi-level (dinosaur) coupled stepper (chronos_esm/coupler/dino_step).

Confirms the dino atmosphere works as THE active coupled model as ONE differentiable
function: a short coupled run stays finite, and value_and_grad of ocean-mean SST w.r.t.
the seed SST flows end-to-end through the modal<->nodal transforms, the lat regridders,
the bulk fluxes and the ocean scan (the P1 differentiability exit criterion).
"""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pooch
import pytest

# DinoCoupledModel builds init_model() (WOA) + DinoAtmosphere(orography) -> ~900 MB
# ETOPO; skip if not cached (CI). Stage with experiments/prefetch_data.py.
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(pooch.os_cache("chronos_esm"), "etopo1.nc")),
    reason="ETOPO1 (~900 MB) not cached; run experiments/prefetch_data.py to enable",
)


@pytest.fixture(scope="module")
def model():
    from chronos_esm.coupler.dino_step import DinoCoupledModel
    return DinoCoupledModel(ocean_ic="woa")


def test_short_run_finite(model):
    """Two short coupling intervals stay finite (ocean, modal atmos, diagnostics)."""
    s = model.init_state()
    for _ in range(2):
        s = model.step(s, interval=0.25)
    assert np.isfinite(np.asarray(s.ocean.temp)).all(), "ocean temp non-finite"
    assert np.isfinite(np.asarray(s.ocean.salt)).all(), "ocean salt non-finite"
    # vorticity is complex spectral coefficients; np.isfinite handles complex.
    assert np.isfinite(np.asarray(s.atmos.vorticity)).all(), "modal vorticity non-finite"
    diag = model.diagnostics_lin(s)
    for k, v in diag.items():
        assert np.isfinite(np.asarray(v)).all(), f"diagnostic {k} non-finite"
    assert s.day == pytest.approx(0.5)


def test_grad_sst_seed_through_interval(model):
    """value_and_grad of ocean-mean SST w.r.t. a uniform seed-SST perturbation,
    through one (short, remat'd) coupling interval, is finite AND non-zero on CPU.
    Non-zero proves the seed flows through the whole coupled path; finite proves the
    modal<->nodal + regrid + flux + ocean-scan graph differentiates without NaN."""
    s0 = model.init_state()
    omask = model.omask

    def ocean_mean_sst(eps):
        temp = s0.ocean.temp.at[0].add(eps)
        s_in = s0._replace(ocean=s0.ocean._replace(temp=temp))
        s_out = model.step(s_in, interval=0.1)
        sst = s_out.ocean.temp[0]
        return jnp.sum(jnp.where(omask, sst, 0.0)) / jnp.sum(omask)

    val, grad = jax.value_and_grad(ocean_mean_sst)(0.0)
    assert np.isfinite(float(val)), f"SST mean non-finite: {val}"
    assert np.isfinite(float(grad)), f"gradient non-finite (graph broken): {grad}"
    # a seed SST bump must change the final mean SST (restoring damps but does not
    # zero it over 0.1 day) -> the coupled graph carries a real, finite sensitivity.
    assert abs(float(grad)) > 1e-4, f"gradient unexpectedly ~0 (graph severed?): {grad}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
