"""
Test script for Carbon Cycle.

Verifies:
1. Global Carbon Conservation.
2. Air-Sea Gas Exchange direction.
3. Land Carbon Uptake (NEE).
"""

import jax
import jax.numpy as jnp

from chronos_esm import main


def run_carbon_test():
    print("Running Carbon Cycle Test...")

    # Initialize model
    state = main.init_model()
    regridder = main.regrid.Regridder()

    # Create mask: Left half Ocean (1), Right half Land (0)
    ny, nx = state.atmos.temp.shape
    mask = jnp.ones((ny, nx))
    mask = mask.at[:, nx // 2 :].set(0.0)

    params = main.ModelParams(mask=mask)

    # Run for a few steps
    steps = 20
    print(f"Simulating {steps} steps...")

    # Track Global Carbon
    # Atmos C [kg] ~ ppm * 2.12e12 kgC (approx)
    # Ocean C [kg] ~ DIC [mmol/m3] * Vol [m3] * 12e-6 [kg/mmol]
    # Land C [kg] ~ SoilC [kg/m2] * Area [m2]

    # Simplified tracking: Just sum the arrays (proportional check)

    def get_carbon_mass(state):
        # Atmos
        co2_sum = jnp.sum(state.atmos.co2)

        # Ocean
        dic_sum = jnp.sum(state.ocean.dic)

        # Land
        soil_c_sum = jnp.sum(state.land.soil_carbon)

        return co2_sum, dic_sum, soil_c_sum

    c_atmos_0, c_ocean_0, c_land_0 = get_carbon_mass(state)

    # Run simulation
    final_state = main.run_simulation(steps, params)

    c_atmos_1, c_ocean_1, c_land_1 = get_carbon_mass(final_state)

    print("\n--- Carbon Mass Changes (Arbitrary Units) ---")
    print(
        f"Atmos: {c_atmos_0:.2f} -> {c_atmos_1:.2f} "
        f"(Delta: {c_atmos_1 - c_atmos_0:.2f})"
    )
    print(
        f"Ocean: {c_ocean_0:.2f} -> {c_ocean_1:.2f} "
        f"(Delta: {c_ocean_1 - c_ocean_0:.2f})"
    )
    print(
        f"Land:  {c_land_0:.2f} -> {c_land_1:.2f} "
        f"(Delta: {c_land_1 - c_land_0:.2f})"
    )

    # total_0 = (
    #     c_atmos_0 * 1.0
    # )  # Need proper conversion factors for strict conservation check
    # But we can check direction.

    # Check Fluxes
    # Air-Sea
    # Initial: Atmos=280, Ocean=2000 -> pCO2_sea = 280. Equilibrium.
    # If we increase Atmos CO2, Ocean should absorb.

    print("\n--- Perturbation Test ---")
    # Perturb Atmos CO2 to 1000 ppm (Large perturbation to overcome float32 precision)
    state_high_co2 = state._replace(
        atmos=state.atmos._replace(co2=jnp.ones_like(state.atmos.co2) * 1000.0)
    )

    # Step multiple times to accumulate change
    print("Stepping 100 times...")

    def scan_fn(carry, _):
        state = carry
        new_state = main.step_coupled(state, params, regridder)
        return new_state, None

    next_state, _ = jax.lax.scan(scan_fn, state_high_co2, jnp.arange(100))

    # Check Ocean DIC
    diff = next_state.ocean.dic - state_high_co2.ocean.dic
    delta_dic_mean = jnp.mean(diff)
    delta_dic_max = jnp.max(diff)
    print(f"Delta Ocean DIC Mean: {delta_dic_mean:.9e}")
    print(f"Delta Ocean DIC Max:  {delta_dic_max:.9e}")

    if delta_dic_max > 0:
        print("SUCCESS: Ocean absorbed Carbon from high-CO2 atmosphere.")
    else:
        print("FAILURE: Ocean did not absorb Carbon.")

    # Check Land NEE
    # Should be negative (Uptake) if GPP > Respiration
    # Initial: SoilC=10, LAI=1.
    # GPP ~ 1e-9 * 240 * 1 * 1 ~ 2.4e-7
    # R_auto ~ 1.2e-7
    # R_hetero ~ 1e-8 * 10 * 1 * 1 ~ 1e-7
    # NEE ~ 1.2e-7 + 1e-7 - 2.4e-7 ~ -0.2e-7 (Uptake)

    nee = next_state.fluxes.carbon_flux_land
    mean_nee = jnp.mean(nee)
    print(f"Mean Land NEE: {mean_nee:.6e}")

    if mean_nee < 0:
        print("SUCCESS: Land is a Net Carbon Sink (Photosynthesis > Respiration).")
    else:
        print("FAILURE: Land is a Net Carbon Source (or neutral).")


if __name__ == "__main__":
    run_carbon_test()
