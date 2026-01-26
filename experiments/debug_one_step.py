
import sys
from pathlib import Path
import jax.numpy as jnp
import jax

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_esm import main, data
from chronos_esm.atmos import physics

def debug_one_step():
    print("Initializing model...")
    state = main.init_model()
    regridder = main.regrid.Regridder()
    mask = data.load_bathymetry_mask()
    params = main.ModelParams(mask=mask, co2_increase_rate=0.0)
    
    # Run ONE step
    print("Running Step 1...")
    
    # We will manually call step_coupled internals to see what happens
    
    # 1. Update CO2 (My new code)
    seconds_per_year = 365.0 * 24.0 * 3600.0
    years_elapsed = state.time / seconds_per_year
    current_co2_ppm = params.co2_ppm * jnp.power(1.0 + params.co2_increase_rate, years_elapsed)
    print(f"Current CO2: {current_co2_ppm}")
    
    forced_atmos = state.atmos._replace(co2=jnp.ones_like(state.atmos.co2) * current_co2_ppm)
    state = state._replace(atmos=forced_atmos)
    
    # 2. Main Step
    # We can just call step_coupled
        # We need to peek inside step_coupled -> step_atmos
        # Let's extract forces manually
    beta = 1.0
    sensible_flux, latent_flux = physics.compute_surface_fluxes(
            state.atmos.temp, state.atmos.q, state.atmos.u, state.atmos.v,
            state.fluxes.sst, beta=beta
    )
    
    # Physics Forces
    q_sat = physics.compute_saturation_humidity(state.atmos.temp, jnp.exp(state.atmos.ln_ps))
    precip, heating = physics.compute_precipitation(state.atmos.q, q_sat, epsilon=1e-3)
    
    co2_mean = 280.0
    lat = jnp.linspace(-90, 90, 96)
    lat_rad = jnp.deg2rad(lat)
    cooling = physics.compute_radiative_forcing(state.atmos.temp, co2_mean, lat_rad)
    
    print("\n--- Diagnostic Forces ---")
    print(f"Sensible Flux (W/m2): Max={jnp.max(jnp.abs(sensible_flux)):.2e}")
    print(f"Latent Flux (W/m2):   Max={jnp.max(jnp.abs(latent_flux)):.2e}")
    print(f"Precip Heating (K/s): Max={jnp.max(heating):.2e}")
    print(f"Rad Cooling (K/s):    Max={jnp.max(jnp.abs(cooling)):.2e}")
    
    # Dynamics Scaling
    CP = 1004.0
    MASS = 1.0e4
    
    term_sensible = sensible_flux / (CP * MASS)
    term_precip = heating
    term_rad = cooling # It is already K/s in physics.py? Let's check physics.py
    
    # Checking physics.py for Radiative Cooling units
    # Newtonian cooling: -rate * (T-Teq) -> [1/s * K] = [K/s]. Correct.
    # CO2 forcing: flux / capacity. Capacity ~ 1e7. Flux ~ W/m2. W/m2 / (J/K/m2) = K/s. Correct.
    
    # Precipitation Heating
    # heating = (Lv/Cp) * precip. precip is [kg/kg/s]?
    # compute_precipitation returns:
    # excess = (q - qc)/eps
    # precip = relu(excess) * eps / tau. -> [kg/kg / s]. Correct.
    # heating = (2.5e6 / 1004) * precip. -> [J/kg / (J/kg/K)] = [K/s]. Correct.
    
    print(f"Term Sensible (K/s):  Max={jnp.max(jnp.abs(term_sensible)):.2e}")
    print(f"Term Precip (K/s):    Max={jnp.max(term_precip):.2e}")
    print(f"Term Rad (K/s):       Max={jnp.max(jnp.abs(term_rad)):.2e}")
    
    total_forcing_t = term_sensible + term_precip + term_rad
    print(f"Total Forcing T (K/s): Max={jnp.max(jnp.abs(total_forcing_t)):.2e}")
    
    dt = 30.0
    dT = total_forcing_t * dt
    print(f"dT per step (K):      Max={jnp.max(jnp.abs(dT)):.2f}")

if __name__ == "__main__":
    debug_one_step()
