"""
Test script for BSTSensor (Biomass / Salinity / Temperature).

Sensor: BSTSensor
Output: [biomass, salinity, temperature]
  - biomass:     relative biomass density at current depth (unitless)
  - salinity:    water salinity at current depth (PSU)
  - temperature: water temperature at current depth (°C)

The sensor models an ocean water column with a thermocline/halocline.

Config:
  surface_temp=20 °C, deep_temp=4 °C
  thermocline_depth=100 m, thermocline_thickness=50 m
  surface_psu=35, deep_psu=34

Agent spawned at z=-50 (50 m depth).
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_bst_sensor",
    "world": "OpenWater",
    "package_name": "Ocean",
    "main_agent": "auv0",
    "ticks_per_sec": 30,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "TorpedoAUV",
            "sensors": [
                {
                    "sensor_type": "BSTSensor",
                    "sensor_name": "BSTSensor",
                    "configuration": {
                        "surface_temp": 20,
                        "deep_temp": 4,
                        "thermocline_depth": 100,
                        "thermocline_thickness": 50,
                        "surface_psu": 35,
                        "deep_psu": 34,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== BSTSensor Test ===")
    print("Output: [biomass, salinity, temperature]")
    print("  biomass:     relative biomass density (unitless)")
    print("  salinity:    PSU")
    print("  temperature: °C")
    print("Spawned at 50 m depth.\n")

    binary_path = holoocean.packagemanager.get_binary_path_for_package("Ocean")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=SCENARIO,
        binary_path=binary_path,
        show_viewport=False,
        verbose=False,
        uuid=str(uuid.uuid4()),
        ticks_per_sec=30,
    ) as env:
        action = np.array([0.0, 0.0, 0.0, 0.0, 60.0])
        for step in range(3000):
            state = env.step(action)

            if step % 50 == 0:
                bst = state.get("BSTSensor", None)
                if bst is not None:
                    arr = np.ravel(bst)
                    biomass = arr[0] if len(arr) > 0 else float("nan")
                    salinity = arr[1] if len(arr) > 1 else float("nan")
                    temperature = arr[2] if len(arr) > 2 else float("nan")
                    print(
                        f"Step {step:4d} | "
                        f"biomass={biomass:8.4f}  "
                        f"salinity={salinity:6.3f} PSU  "
                        f"temp={temperature:6.3f} °C"
                    )
