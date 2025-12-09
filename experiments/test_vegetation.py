"""
Test script for Dynamic Vegetation.

Runs a single-point simulation to verify:
1. Vegetation growth (LAI increase) under favorable conditions.
2. Dynamic Albedo change (Soil -> Vegetation).
3. Seasonal cycle response (if we vary temp).
"""

import jax
import jax.numpy as jnp

from chronos_esm.config import DT_ATMOS
from chronos_esm.land import vegetation


def run_point_test():
    print("Running Vegetation Point Test...")

    # Setup single point
    # ny, nx = 1, 1

    # Initial state: Bare soil, some moisture
    temp = jnp.array([[295.0]])  # 22C, good for growth
    soil_moisture = jnp.array([[0.1]])  # 10cm, healthy
    lai = jnp.array([[0.1]])  # Bare soil start

    # Constants
    BUCKET_DEPTH = 0.15

    # Simulation loop
    days = 100
    steps = int(days * 86400 / DT_ATMOS)

    lai_history = []
    albedo_history = []

    print(f"Simulating {days} days ({steps} steps)...")

    current_lai = lai

    # JIT the step function for speed
    @jax.jit
    def step_fn(lai, temp, moisture):
        lai_new = vegetation.step_vegetation(lai, temp, moisture, BUCKET_DEPTH)
        albedo, _ = vegetation.compute_land_properties(lai_new)
        return lai_new, albedo

    for i in range(steps):
        # Update vegetation
        current_lai, current_albedo = step_fn(current_lai, temp, soil_moisture)

        # Store (convert to numpy for list)
        lai_history.append(float(current_lai[0, 0]))
        albedo_history.append(float(current_albedo[0, 0]))

        # Simple scenario:
        # Day 0-50: Good conditions (Temp=295, Moisture=0.1)
        # Day 50-100: Drought (Moisture=0.0)
        if i > steps // 2:
            soil_moisture = jnp.array([[0.0]])

    # Results
    lai_start = lai_history[0]
    lai_peak = lai_history[steps // 2]
    lai_end = lai_history[-1]

    print(f"LAI Start: {lai_start:.4f}")
    print(f"LAI Peak (Day 50): {lai_peak:.4f}")
    print(f"LAI End (Day 100): {lai_end:.4f}")

    if lai_peak > lai_start:
        print("SUCCESS: Vegetation grew under favorable conditions.")
    else:
        print("FAILURE: Vegetation did not grow.")

    if lai_end < lai_peak:
        print("SUCCESS: Vegetation decayed under drought.")
    else:
        print("FAILURE: Vegetation did not decay.")

    # Albedo check
    alb_start = albedo_history[0]
    alb_peak = albedo_history[steps // 2]
    print(f"Albedo Start: {alb_start:.4f}")
    print(f"Albedo Peak: {alb_peak:.4f}")

    if alb_peak < alb_start:
        print("SUCCESS: Albedo decreased (darkened) as vegetation grew.")
    else:
        print("FAILURE: Albedo did not decrease.")


if __name__ == "__main__":
    run_point_test()
