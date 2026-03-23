"""
Test script for RotationSensor.

Sensor: RotationSensor
Output: [roll, pitch, yaw] in degrees.
  - roll:  rotation around forward axis
  - pitch: rotation around lateral axis (nose up/down)
  - yaw:   rotation around vertical axis (heading)
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_rotation_sensor",
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
                    "sensor_type": "RotationSensor",
                    "sensor_name": "RotationSensor",
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== RotationSensor Test ===")
    print("Output: [roll, pitch, yaw] in degrees.\n")

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
                rot = state.get("RotationSensor", None)
                if rot is not None:
                    roll, pitch, yaw = rot[0], rot[1], rot[2]
                    print(f"Step {step:4d} | roll={roll:8.3f}°  pitch={pitch:8.3f}°  yaw={yaw:8.3f}°")
