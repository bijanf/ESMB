
import sys
import jax.numpy as jnp
from chronos_esm import data
from chronos_esm.atmos import dynamics as atmos_driver

def check_ic():
    print("Checking Ocean ICs...")
    temp_ic, salt_ic = data.load_initial_conditions(nz=15)
    print(f"Ocean Temp IC Mean (raw): {jnp.mean(temp_ic):.2f}")
    print(f"Ocean Temp IC Min: {jnp.min(temp_ic):.2f}")
    print(f"Ocean Temp IC Max: {jnp.max(temp_ic):.2f}")
    
    print("\nChecking Atmos ICs...")
    atmos = atmos_driver.init_atmos_state(48, 96)
    print(f"Atmos Temp IC Mean: {jnp.mean(atmos.temp):.2f}")
    print(f"Atmos Temp IC Min: {jnp.min(atmos.temp):.2f}")
    print(f"Atmos Temp IC Max: {jnp.max(atmos.temp):.2f}")

if __name__ == "__main__":
    check_ic()
