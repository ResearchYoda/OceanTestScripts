"""
Test script for LocationSensor.

Sensor: LocationSensor
Output: [x, y, z] position of the agent in world coordinates (meters).
  - x: forward/backward
  - y: left/right
  - z: up/down (negative = below surface)
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_location_sensor",
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
                    "sensor_type": "LocationSensor",
                    "sensor_name": "LocationSensor",
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== LocationSensor Test ===")
    print("Output: [x, y, z] world position in meters.")
    print("Negative z = below surface.\n")

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
                loc = state.get("LocationSensor", None)
                if loc is not None:
                    x, y, z = loc[0], loc[1], loc[2]
                    print(f"Step {step:4d} | x={x:8.3f}  y={y:8.3f}  z={z:8.3f} m")
