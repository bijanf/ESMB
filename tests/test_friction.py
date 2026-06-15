
import os
import sys
import jax.numpy as jnp
import numpy as np

sys.path.append(os.getcwd())
import pooch  # noqa: E402
import pytest  # noqa: E402

from chronos_esm import main, data
from chronos_esm.config import DT_OCEAN

# init_model() pulls the ~900 MB ETOPO bathymetry; skip if it isn't cached (e.g. in
# CI). Stage it with `python experiments/prefetch_data.py` to run this test.
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(pooch.os_cache("chronos_esm"), "etopo1.nc")),
    reason="ETOPO1 (~900 MB) not cached; run experiments/prefetch_data.py to enable",
)


def test_friction():
    print("Running Friction Test...")
    mask = jnp.array(data.load_bathymetry_mask())
    params = main.ModelParams(mask=mask)
    
    # Run for 20 steps (enough to spin up intial geostrophy from density)
    print("Simulating 20 steps...")
    state = main.init_model()
    regridder = main.regrid.Regridder()
    
    # Manually step to check velocities
    steps = 20
    for i in range(steps):
        state = main.step_coupled(state, params, regridder)
        u_max = float(jnp.max(jnp.abs(state.ocean.u)))
        v_max = float(jnp.max(jnp.abs(state.ocean.v)))
        print(f"Step {i+1}: Max U={u_max:.4f}, Max V={v_max:.4f} m/s")
        
        if u_max > 5.0 or v_max > 5.0:
            print("FAILURE: Stability Check should have triggered (check if NaNs appeared)")
            if jnp.isnan(u_max):
                print("SUCCESS: Model crashed with NaNs as expected (if it was unstable).")
                break
    
    print("\nFinal Velocity Check:")
    print(f"Max U: {u_max} m/s")
    if u_max < 0.5:
        print("SUCCESS: Friction is effective. Velocity is < 0.5 m/s.")
    else:
        print("WARNING: Velocity is still high. Friction might be too weak.")

if __name__ == "__main__":
    test_friction()
