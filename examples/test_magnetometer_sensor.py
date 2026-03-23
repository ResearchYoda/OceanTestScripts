"""
Test script for MagnetometerSensor.

Sensor: MagnetometerSensor
Output: [mx, my, mz] — magnetic field vector in body frame (normalized, unitless).
  The vector points toward magnetic north relative to the agent's orientation.
  With Sigma noise applied.

Config: Sigma=0.01 (Gaussian noise on each component).
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_magnetometer_sensor",
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
                    "sensor_type": "MagnetometerSensor",
                    "sensor_name": "MagnetometerSensor",
                    "configuration": {
                        "Sigma": 0.01,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== MagnetometerSensor Test ===")
    print("Output: [mx, my, mz] magnetic field vector in body frame.")
    print("Config: Sigma=0.01 noise.\n")

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
                mag = state.get("MagnetometerSensor", None)
                if mag is not None:
                    arr = np.ravel(mag)
                    mx, my, mz = arr[0], arr[1], arr[2]
                    norm = np.linalg.norm(arr[:3])
                    print(f"Step {step:4d} | mx={mx:7.4f}  my={my:7.4f}  mz={mz:7.4f}  |m|={norm:.4f}")
