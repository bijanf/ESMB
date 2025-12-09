import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import uvicorn
from fastapi import BackgroundTasks, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from chronos_esm.coupler import regrid
from chronos_esm.data import load_bathymetry_mask
from chronos_esm.main import ModelParams, init_model, step_coupled
from chronos_esm.ocean import diagnostics as ocean_diagnostics

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global Simulation State
class SimulationRunner:
    def __init__(self):
        self.state = None
        self.params = ModelParams()
        self.regridder = None
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        self.step_count = 0
        self.target_year = 2025.0  # Default target year (1 year run)
        self.BASE_YEAR = 2024.0

    def initialize(self):
        print("Initializing model...")

        # Load realistic Land Mask from WOA18 data
        print("Loading realistic land mask...")
        try:
            mask_bool = load_bathymetry_mask()
            # Convert boolean (True=Ocean) to float (1.0=Ocean, 0.0=Land)
            mask = mask_bool.astype(float)
            print("Realistic mask loaded.")
        except Exception as e:
            print(
                f"Failed to load realistic mask: {e}. Falling back to simple continents."
            )
            # Fallback to simple continents if download fails
            mask = jnp.ones((96, 192))
            mask = mask.at[:, 60:90].set(0.0)
            mask = mask.at[:, 140:170].set(0.0)
            mask = mask.at[:5, :].set(1.0)
            mask = mask.at[-5:, :].set(1.0)

        self.params = ModelParams(mask=mask)

        self.state = init_model()
        self.regridder = regrid.Regridder()
        self.step_count = 0
        # Force compilation
        print("Compiling step function...")
        _ = step_coupled(self.state, self.params, self.regridder)
        print("Model initialized.")

    def step(self):
        if self.state is None:
            return

        with self.lock:
            self.state = step_coupled(self.state, self.params, self.regridder)
            self.step_count += 1
            # Block to ensure computation is done (optional, but good for timing)
            # self.state.atmos.temp.block_until_ready()

    def run_loop(self):
        STEPS_PER_BATCH = (
            2000  # Run 2000 steps per UI update (approx 8 hours per frame at dt=15)
        )

        while self.is_running:
            # Check for target year
            current_year = self.BASE_YEAR + float(self.state.time) / (3600 * 24 * 360)
            if self.target_year is not None and current_year >= self.target_year:
                print(f"Target year {self.target_year} reached. Stopping.")
                self.is_running = False
                break

            # Batch stepping
            for _ in range(STEPS_PER_BATCH):
                if not self.is_running:
                    break
                self.step()

            # Small sleep to yield thread
            time.sleep(0.001)

    def start(self):
        if self.state is None:
            self.initialize()

        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self.run_loop)
            self.thread.start()

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
            self.thread = None

    def reset(self):
        self.stop()
        self.initialize()

    def update_params(self, new_params: dict):
        # Handle target_year separately as it's not in ModelParams
        if "target_year" in new_params:
            self.target_year = new_params.pop("target_year")
            print(f"Updated target_year: {self.target_year}")

        # Update specific fields in ModelParams
        if new_params:
            current_dict = self.params._asdict()
            current_dict.update(new_params)
            self.params = ModelParams(**current_dict)
            print(f"Updated parameters: {self.params}")

    def get_snapshot(self):
        if self.state is None:
            return None

        with self.lock:
            # Extract fields and convert to numpy (CPU)
            # Downsample for visualization if needed, but 96x192 is small enough

            # Atmos
            temp_atm = np.array(self.state.atmos.temp)
            vort_atm = np.array(self.state.atmos.vorticity)
            co2_atm = np.array(self.state.atmos.co2)
            u_atm = np.array(self.state.atmos.u)
            v_atm = np.array(self.state.atmos.v)

            # Fluxes
            precip = np.array(self.state.fluxes.precip)

            # Ocean (Surface)
            temp_ocn = np.array(self.state.ocean.temp[0])  # Surface

            # Land
            temp_land = np.array(self.state.land.temp)

            # Metrics
            global_temp = float(np.mean(temp_atm))
            amoc_index = float(ocean_diagnostics.compute_amoc_index(self.state.ocean))

            # Time in years (approx)
            current_year = self.BASE_YEAR + float(self.state.time) / (3600 * 24 * 360)

            # Helper to sanitize NaNs
            def sanitize(arr):
                return np.where(np.isnan(arr), 0.0, arr).tolist()

            return {
                "step": self.step_count,
                "time": float(self.state.time),
                "year": current_year,
                "global_temp": global_temp if not np.isnan(global_temp) else 0.0,
                "amoc_index": amoc_index if not np.isnan(amoc_index) else 0.0,
                "params": {**self.params._asdict(), "target_year": self.target_year},
                "fields": {
                    "temp_atm": sanitize(temp_atm),
                    "vort_atm": sanitize(vort_atm),
                    "co2_atm": sanitize(co2_atm),
                    "u_atm": sanitize(u_atm),
                    "v_atm": sanitize(v_atm),
                    "precip": sanitize(precip),
                    "temp_ocn": sanitize(temp_ocn),
                    "temp_land": sanitize(temp_land),
                },
            }


runner = SimulationRunner()


class UpdateParamsRequest(BaseModel):
    co2_ppm: Optional[float] = None
    solar_constant: Optional[float] = None
    target_year: Optional[float] = None


@app.on_event("startup")
async def startup_event():
    runner.initialize()


@app.post("/start")
async def start_simulation():
    runner.start()
    return {"status": "started"}


@app.post("/stop")
async def stop_simulation():
    runner.stop()
    return {"status": "stopped"}


@app.post("/reset")
async def reset_simulation():
    runner.reset()
    return {"status": "reset"}


@app.post("/step")
async def step_simulation():
    runner.step()
    return {"status": "stepped", "step": runner.step_count}


@app.post("/update_params")
async def update_params(request: UpdateParamsRequest):
    params_to_update = {k: v for k, v in request.dict().items() if v is not None}
    runner.update_params(params_to_update)
    return {"status": "updated", "params": runner.params._asdict()}


@app.get("/state")
async def get_state():
    snapshot = runner.get_snapshot()
    if snapshot is None:
        return {"status": "not_initialized"}
    return snapshot


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
