"""
Test script for PoseSensor.

Sensor: PoseSensor
Output: 4x4 SE3 homogeneous transformation matrix.
  - Upper-left 3x3: rotation matrix (body orientation in world frame)
  - Right column [0:3, 3]: translation (world position in meters)
  - Bottom row: [0, 0, 0, 1]
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_pose_sensor",
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
                    "sensor_type": "PoseSensor",
                    "sensor_name": "PoseSensor",
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== PoseSensor Test ===")
    print("Output: 4x4 SE3 homogeneous transformation matrix.")
    print("  Upper-left 3x3 = rotation, right column [0:3] = position (m).\n")

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
                pose = state.get("PoseSensor", None)
                if pose is not None:
                    print(f"Step {step:4d} | SE3 matrix:")
                    for row in pose:
                        print("  " + "  ".join(f"{v:9.4f}" for v in row))
                    print()
