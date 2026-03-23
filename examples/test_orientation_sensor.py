"""
Test script for OrientationSensor.

Sensor: OrientationSensor
Output: 3x3 rotation matrix describing the body orientation in the world frame.
  - Row 0: forward direction vector in world frame
  - Row 1: right direction vector in world frame
  - Row 2: up direction vector in world frame
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_orientation_sensor",
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
                    "sensor_type": "OrientationSensor",
                    "sensor_name": "OrientationSensor",
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== OrientationSensor Test ===")
    print("Output: 3x3 rotation matrix.")
    print("  Row 0 = forward vector, Row 1 = right vector, Row 2 = up vector.\n")

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

            if step % 100 == 0:
                ori = state.get("OrientationSensor", None)
                if ori is not None:
                    print(f"Step {step:4d} | Rotation matrix (forward/right/up):")
                    labels = ["forward", "right  ", "up     "]
                    for label, row in zip(labels, ori):
                        print(f"  {label}: [{row[0]:8.4f}  {row[1]:8.4f}  {row[2]:8.4f}]")
                    print()
