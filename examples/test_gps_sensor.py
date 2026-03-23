"""
Test script for GPSSensor.

Sensor: GPSSensor
Output: [x, y, z] position (m) or zeros/None when agent is too deep.
  The sensor only returns valid readings when the agent is within Depth meters of
  the surface. Beyond that the GPS signal is unavailable.

Config: Sigma=0.5 (horizontal noise, m), Depth=1 (max depth for valid fix, m),
        DepthSigma=0.25 (depth noise, m).

Agent spawned near the surface at z=-1.
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_gps_sensor",
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
                    "sensor_type": "GPSSensor",
                    "sensor_name": "GPSSensor",
                    "configuration": {
                        "Sigma": 0.5,
                        "Depth": 1,
                        "DepthSigma": 0.25,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== GPSSensor Test ===")
    print("Output: [x, y, z] position (m). Only valid when within 1 m of surface.")
    print("Config: Sigma=0.5 m, Depth=1 m, DepthSigma=0.25 m. Spawned at z=-1.\n")

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
                gps = state.get("GPSSensor", None)
                if gps is not None:
                    arr = np.ravel(gps)
                    if len(arr) >= 3:
                        x, y, z = arr[0], arr[1], arr[2]
                        mag = np.linalg.norm(arr[:3])
                        valid = "valid" if mag > 1e-6 else "no fix (too deep)"
                        print(f"Step {step:4d} | x={x:8.3f}  y={y:8.3f}  z={z:8.3f}  [{valid}]")
                    else:
                        print(f"Step {step:4d} | GPS data: {arr}")
                else:
                    print(f"Step {step:4d} | GPS: no data")
