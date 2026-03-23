"""
Test script for CollisionSensor.

Sensor: CollisionSensor
Output: [bool] — 1.0 (or True) if the agent is currently colliding, 0.0 otherwise.
  Useful for detecting contact with environment geometry or other agents.
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_collision_sensor",
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
                    "sensor_type": "CollisionSensor",
                    "sensor_name": "CollisionSensor",
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== CollisionSensor Test ===")
    print("Output: [bool] — 1 if colliding, 0 otherwise.\n")

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
                col = state.get("CollisionSensor", None)
                if col is not None:
                    arr = np.ravel(col)
                    colliding = bool(arr[0])
                    status = "COLLIDING" if colliding else "not colliding"
                    print(f"Step {step:4d} | CollisionSensor = {int(colliding)}  [{status}]")
