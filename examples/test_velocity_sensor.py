"""
Test script for VelocitySensor.

Sensor: VelocitySensor
Output: [vx, vy, vz] velocity in world frame (m/s).
  - vx: velocity along world X axis
  - vy: velocity along world Y axis
  - vz: velocity along world Z axis (positive = upward)
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_velocity_sensor",
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
                    "sensor_type": "VelocitySensor",
                    "sensor_name": "VelocitySensor",
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== VelocitySensor Test ===")
    print("Output: [vx, vy, vz] velocity in world frame (m/s).\n")

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
                vel = state.get("VelocitySensor", None)
                if vel is not None:
                    vx, vy, vz = vel[0], vel[1], vel[2]
                    speed = np.linalg.norm(vel[:3])
                    print(f"Step {step:4d} | vx={vx:7.4f}  vy={vy:7.4f}  vz={vz:7.4f}  |v|={speed:.4f} m/s")
