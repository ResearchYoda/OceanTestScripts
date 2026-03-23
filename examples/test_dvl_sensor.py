"""
Test script for DVLSensor.

Sensor: DVLSensor (Doppler Velocity Log)
Output (ReturnRange=True): 7-element array
  [0:3]  [vx, vy, vz]  — velocity in body frame (m/s)
  [3:7]  [range_xf, range_yf, range_xb, range_yb] — slant ranges to seabed from
         four beams (forward-X, forward-Y, back-X, back-Y) in meters.

Config:
  Elevation=22.5 deg (beam elevation angle from horizontal)
  VelSigma=0.02626, ReturnRange=True, MaxRange=50 m, RangeSigma=0.1 m
Socket: DVLSocket
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_dvl_sensor",
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
                    "sensor_type": "DVLSensor",
                    "sensor_name": "DVLSensor",
                    "socket": "DVLSocket",
                    "configuration": {
                        "Elevation": 22.5,
                        "VelSigma": 0.02626,
                        "ReturnRange": True,
                        "MaxRange": 50,
                        "RangeSigma": 0.1,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== DVLSensor Test ===")
    print("Output: [vx, vy, vz, range_xf, range_yf, range_xb, range_yb]")
    print("  vx/vy/vz: body-frame velocity (m/s)")
    print("  ranges: beam slant ranges to seabed (m), -1 if beyond MaxRange.\n")

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
                dvl = state.get("DVLSensor", None)
                if dvl is not None:
                    arr = np.ravel(dvl)
                    vx, vy, vz = arr[0], arr[1], arr[2]
                    ranges = arr[3:7] if len(arr) >= 7 else [float("nan")] * 4
                    print(
                        f"Step {step:4d} | "
                        f"vel=({vx:7.4f}, {vy:7.4f}, {vz:7.4f}) m/s  "
                        f"ranges=({ranges[0]:.2f}, {ranges[1]:.2f}, {ranges[2]:.2f}, {ranges[3]:.2f}) m"
                    )
