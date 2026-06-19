"""Ocean SST flux-correction: strong Haney restoring (control) vs frozen q-flux +
weak anomaly restoring (free, forcing-responsive). Working-ESM P2.

The strong WOA restoring (tau~30 d, lambda~50 W/m^2/K) holds the mean state but
ABSORBS any imposed forcing, so the coupled model cannot respond to CO2. The q-flux
("mixed-layer flux correction") method frees it without re-opening the cold-tongue drift:

  1. run a strongly-restored CONTROL to equilibrium and accumulate the time-mean
     restoring heat flux  Q_bar = <lambda*(SST_target - SST)>  -- the *implied* flux
     correction that the restoring was silently supplying to hold the mean state;
  2. in the FREE run, PRESCRIBE Q_bar (a fixed field) plus only a WEAK long-tau anomaly
     restoring. The mean state is held by Q_bar, drift is bounded by the weak term, but
     SST is now free to respond to an imposed forcing (CO2) on the anomaly timescale.

All pure jnp -> differentiable (d(climate)/d(forcing) survives the flux correction).
"""
import jax.numpy as jnp

from chronos_esm.config import RHO_WATER, CP_WATER, OCEAN_DZ


def restoring_lambda(tau_days, dz_surf=float(OCEAN_DZ[0])):
    """Haney restoring coefficient lambda [W/m^2/K] = rho*cp*dz_surf / tau."""
    return RHO_WATER * CP_WATER * dz_surf / (tau_days * 86400.0)


def restoring_flux(sst_K, sst_target_K, tau_days):
    """Haney restoring heat flux [W/m^2] = lambda*(target - SST) (positive = into ocean
    where it is colder than target). Its time-mean over a converged control IS the q-flux."""
    return restoring_lambda(tau_days) * (jnp.asarray(sst_target_K) - sst_K)


def heat_correction(sst_K, sst_target_K, restore_tau_days, q_flux=None):
    """Surface heat-budget correction [W/m^2] to add to the bulk net heat flux.

    q_flux is None  -> pure (strong) Haney restoring  [CONTROL mode];
    q_flux provided -> prescribed frozen q-flux + a (weak, long-tau) anomaly restoring
                       [FREE / forcing-responsive mode].
    """
    corr = restoring_flux(sst_K, sst_target_K, restore_tau_days)
    if q_flux is not None:
        corr = corr + jnp.asarray(q_flux)
    return corr


__all__ = ["restoring_lambda", "restoring_flux", "heat_correction"]
