"""
Test script for RangeFinderSensor.

Sensor: RangeFinderSensor
Output: [distance_m] — distance to nearest obstacle along the sensor ray (meters).
  Returns MaxDistance if no hit within range.

Config:
  LaserMaxDistance=50 m
  LaserCount=1 (single beam)
  LaserAngle=0 (straight ahead)
  LaserDebug=False
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_rangefinder_sensor",
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
                    "sensor_type": "RangeFinderSensor",
                    "sensor_name": "RangeFinderSensor",
                    "configuration": {
                        "LaserMaxDistance": 50,
                        "LaserCount": 1,
                        "LaserAngle": 0,
                        "LaserDebug": False,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== RangeFinderSensor Test ===")
    print("Output: [distance_m] — range to nearest obstacle (m).")
    print("Config: single beam, max 50 m, straight ahead.\n")

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
                rng = state.get("RangeFinderSensor", None)
                if rng is not None:
                    arr = np.ravel(rng)
                    dist = arr[0]
                    hit = "HIT" if dist < 50.0 else "no hit"
                    print(f"Step {step:4d} | distance = {dist:.4f} m  [{hit}]")
