"""
Test script for DepthSensor.

Sensor: DepthSensor
Output: [depth_m] - depth below the water surface in meters (positive = deeper).
Config: Sigma=0.255 (Gaussian noise standard deviation in meters).

Agent spawned at z=-20 (20 m depth).
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_depth_sensor",
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
                    "sensor_type": "DepthSensor",
                    "sensor_name": "DepthSensor",
                    "configuration": {
                        "Sigma": 0.255,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== DepthSensor Test ===")
    print("Output: [depth_m] — depth below water surface (m), positive = deeper.")
    print("Config: Sigma=0.255 m noise. Spawned at 20 m depth.\n")

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
                depth_data = state.get("DepthSensor", None)
                if depth_data is not None:
                    depth = float(np.ravel(depth_data)[0])
                    print(f"Step {step:4d} | depth = {depth:.4f} m")
