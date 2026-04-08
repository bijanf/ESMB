"""
JCM (JAX Climate Model) adapter for Chronos-ESM.

Replaces the custom FFT+FD atmospheric dynamics with JCM/dinosaur,
which uses full spherical harmonics and IMEX RK3 time stepping.
This eliminates polar convergence issues and produces realistic
atmospheric circulation (jet streams, Hadley cell, trade winds).

JCM is based on SPEEDY physics with 8 sigma levels.
Dynamics are handled by Google's dinosaur library.
"""

import jax
import jax.numpy as jnp
import jax_datetime as jdt
import logging

import jcm
from jcm.model import Model
from jcm.forcing import ForcingData, default_forcing
from jcm.diffusion import DiffusionFilter
from jcm.physics.speedy.speedy_coords import get_speedy_coords
from jcm.physics_interface import dynamics_state_to_physics_state
from jcm.terrain import TerrainData

from chronos_esm.config import DT_OCEAN

logger = logging.getLogger(__name__)


class JCMAtmosphere:
    """Wrapper around JCM Model for use in Chronos-ESM coupling.

    Handles initialization, state management, and conversion between
    JCM's internal representation and Chronos-ESM's coupling interface.
    """

    def __init__(
        self,
        time_step_minutes: float = 15.0,
        spectral_truncation: int = 31,
        terrain: TerrainData = None,
        diffusion: DiffusionFilter = None,
        start_date: str = "2000-01-01",
    ):
        """Initialize JCM atmosphere model.

        Args:
            time_step_minutes: Atmosphere timestep in minutes (default 15).
                Must evenly divide DT_OCEAN (900s = 15 min).
            spectral_truncation: Spectral truncation number (31 for T31).
            terrain: Optional TerrainData for topography. None = aquaplanet.
            diffusion: Optional DiffusionFilter. None = default.
            start_date: Start date string for the simulation.
        """
        self.coords = get_speedy_coords(spectral_truncation=spectral_truncation)
        self.nodal_shape_2d = self.coords.nodal_shape[1:]  # (nlon, nlat) e.g. (96, 48)
        self.nodal_shape_3d = self.coords.nodal_shape       # (nlevs, nlon, nlat) e.g. (8, 96, 48)
        self.n_levels = self.coords.nodal_shape[0]

        self.model = Model(
            coords=self.coords,
            time_step=time_step_minutes,
            terrain=terrain,
            diffusion=diffusion or DiffusionFilter.default(),
            start_date=jdt.to_datetime(start_date),
            log_level=logging.WARNING,
        )

        # Compute how many JCM steps fit in one ocean timestep
        jcm_dt_seconds = time_step_minutes * 60.0
        self.steps_per_ocean = int(DT_OCEAN / jcm_dt_seconds)
        assert self.steps_per_ocean >= 1, (
            f"JCM timestep ({jcm_dt_seconds}s) must be <= DT_OCEAN ({DT_OCEAN}s)"
        )

        # Time parameters for run_from_state (in days)
        self.save_interval_days = DT_OCEAN / 86400.0
        self.total_time_days = DT_OCEAN / 86400.0

        # Internal state (spectral/modal space)
        self._modal_state = None
        self._forcing = None

        logger.info(
            "JCM Atmosphere initialized: T%d, dt=%g min, %d steps/ocean_step, "
            "grid=(%d levels, %d lon, %d lat)",
            spectral_truncation, time_step_minutes, self.steps_per_ocean,
            *self.nodal_shape_3d,
        )

    def initialize(self, sst=None):
        """Initialize JCM state and forcing.

        Args:
            sst: Optional initial SST field (nlon, nlat) in Kelvin.
                 If None, uses JCM's default aquaplanet SST.

        Returns:
            Dictionary with surface diagnostics from initial state.
        """
        self._modal_state = self.model._prepare_initial_modal_state()

        # Set up forcing
        self._forcing = default_forcing(self.coords.horizontal)

        if sst is not None:
            # Transpose if needed: Chronos uses (nlat, nlon), JCM uses (nlon, nlat)
            sst_jcm = self._chronos_to_jcm_2d(sst)
            self._forcing = self._forcing.copy(sea_surface_temperature=sst_jcm)

        # Get initial physics state for diagnostics
        physics_state = dynamics_state_to_physics_state(
            self._modal_state, self.model.primitive
        )

        return self._extract_surface_fields(physics_state)

    def step(self, sst, land_temp=None, ice_fraction=None, snow_cover=None,
             soil_moisture=None):
        """Run JCM for one ocean timestep and return surface fluxes.

        Args:
            sst: Sea surface temperature (nlat, nlon) in Kelvin.
            land_temp: Land surface temperature (nlat, nlon) in Kelvin. Optional.
            ice_fraction: Sea ice concentration (nlat, nlon) [0-1]. Optional.
            snow_cover: Snow cover (nlat, nlon) [m]. Optional.
            soil_moisture: Soil moisture (nlat, nlon) [fraction]. Optional.

        Returns:
            Dictionary with:
                - u_surf: Surface zonal wind (nlat, nlon) [m/s]
                - v_surf: Surface meridional wind (nlat, nlon) [m/s]
                - temp_surf: Near-surface temperature (nlat, nlon) [K]
                - q_surf: Near-surface specific humidity (nlat, nlon) [g/kg]
                - precip: Total precipitation (nlat, nlon) [kg/m^2/s]
                - heat_flux: Net surface heat flux (nlat, nlon) [W/m^2], upward positive
                - freshwater_flux: Net freshwater flux (nlat, nlon) [kg/m^2/s]
                - evaporation: Surface evaporation (nlat, nlon) [kg/m^2/s]
        """
        if self._modal_state is None:
            raise RuntimeError("Must call initialize() before step()")

        # Update forcing with current surface conditions
        sst_jcm = self._chronos_to_jcm_2d(sst)
        force_updates = dict(sea_surface_temperature=sst_jcm)

        if land_temp is not None:
            force_updates["stl_am"] = self._chronos_to_jcm_2d(land_temp)
        if ice_fraction is not None:
            force_updates["sice_am"] = self._chronos_to_jcm_2d(ice_fraction)
        if snow_cover is not None:
            force_updates["snowc_am"] = self._chronos_to_jcm_2d(snow_cover)
        if soil_moisture is not None:
            force_updates["soilw_am"] = self._chronos_to_jcm_2d(soil_moisture)

        self._forcing = self._forcing.copy(**force_updates)

        # Run JCM for one ocean timestep
        # output_averages=False is much faster to JIT-compile
        new_modal_state, predictions = self.model.run_from_state(
            initial_state=self._modal_state,
            forcing=self._forcing,
            save_interval=self.save_interval_days,
            total_time=self.total_time_days,
            output_averages=False,
        )

        self._modal_state = new_modal_state

        # Extract physics diagnostics (last timestep snapshot)
        # With output_averages=False, physics has a time dimension [0]
        physics_diag = jax.tree.map(lambda x: x[0] if x.ndim > 0 else x, predictions.physics)

        # Extract surface fields
        physics_state = dynamics_state_to_physics_state(
            new_modal_state, self.model.primitive
        )
        surface = self._extract_surface_fields(physics_state)

        # Extract fluxes from SPEEDY physics diagnostics
        fluxes = self._extract_fluxes(physics_diag)

        return {**surface, **fluxes}

    def _extract_surface_fields(self, physics_state):
        """Extract surface-level fields from JCM PhysicsState.

        JCM's level index 0 is top-of-atmosphere, last index is surface.
        """
        # Bottom level (surface) -- JCM: level index -1 (or n_levels-1)
        # JCM shape: (nlevs, nlon, nlat) -- need to transpose to (nlat, nlon) for Chronos
        u_surf = self._jcm_to_chronos_2d(physics_state.u_wind[-1])
        v_surf = self._jcm_to_chronos_2d(physics_state.v_wind[-1])
        temp_surf = self._jcm_to_chronos_2d(physics_state.temperature[-1])
        q_surf = self._jcm_to_chronos_2d(physics_state.specific_humidity[-1])

        return {
            "u_surf": u_surf,
            "v_surf": v_surf,
            "temp_surf": temp_surf,
            "q_surf": q_surf,
        }

    def _extract_fluxes(self, physics_diag):
        """Extract surface fluxes from SPEEDY physics diagnostics.

        SPEEDY surface_flux fields have a trailing dimension for surface types
        (ocean=0, land=1, ice=2). We sum over surface types to get total.
        hfluxn has 2 components. precnv/precls are already 2D (nlon, nlat).
        u0/v0 are 2D surface diagnostic winds.
        """
        # Total heat flux: sum over 2 components, upward positive
        # hfluxn shape: (nlon, nlat, 2)
        heat_flux = -jnp.sum(physics_diag.surface_flux.hfluxn, axis=-1)
        heat_flux = self._jcm_to_chronos_2d(heat_flux)

        # Evaporation: sum over 3 surface types, upward positive
        # evap shape: (nlon, nlat, 3), units: g/m^2/s
        evaporation = jnp.sum(physics_diag.surface_flux.evap, axis=-1)
        evaporation = self._jcm_to_chronos_2d(evaporation) / 1000.0  # g/m^2/s -> kg/m^2/s

        # Precipitation: already 2D (nlon, nlat), units: g/m^2/s
        precip_conv = self._jcm_to_chronos_2d(physics_diag.convection.precnv) / 1000.0
        precip_ls = self._jcm_to_chronos_2d(physics_diag.condensation.precls) / 1000.0
        precip_total = precip_conv + precip_ls

        # Freshwater flux = evaporation - precipitation (upward positive)
        freshwater_flux = evaporation - precip_total

        # Surface diagnostic winds from SPEEDY (already 2D)
        u0 = self._jcm_to_chronos_2d(physics_diag.surface_flux.u0)
        v0 = self._jcm_to_chronos_2d(physics_diag.surface_flux.v0)

        return {
            "heat_flux": heat_flux,
            "evaporation": evaporation,
            "precip": precip_total,
            "freshwater_flux": freshwater_flux,
            "u0_diag": u0,   # SPEEDY diagnostic surface wind
            "v0_diag": v0,
        }

    def _chronos_to_jcm_2d(self, field):
        """Convert 2D field from Chronos (nlat, nlon) to JCM (nlon, nlat)."""
        return jnp.asarray(field).T

    def _jcm_to_chronos_2d(self, field):
        """Convert 2D field from JCM (nlon, nlat) to Chronos (nlat, nlon)."""
        return jnp.asarray(field).T

    def get_modal_state(self):
        """Return the current modal state for checkpointing."""
        return self._modal_state

    def set_modal_state(self, modal_state):
        """Restore modal state from a checkpoint."""
        self._modal_state = modal_state
