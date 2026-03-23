"""
Test script for RaycastLidar.

Sensor: RaycastLidar
Output: flat 1D array. Reshape to (-1, 5) for columns:
  [0] x         — point x coordinate (m)
  [1] y         — point y coordinate (m)
  [2] z         — point z coordinate (m)
  [3] intensity — return intensity (0-1)
  [4] ring      — laser ring/channel index (0 to Channels-1)

Config:
  Channels=32, Range=50 m
  PointsPerSecond=56000, RotationFrequency=10 Hz
  UpperFovLimit=10 deg, LowerFovLimit=-30 deg

Prints number of points returned and mean range every 50 steps.
No cv2 display.
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_raycast_lidar",
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
                    "sensor_type": "RaycastLidar",
                    "sensor_name": "RaycastLidar",
                    "configuration": {
                        "Channels": 32,
                        "Range": 50,
                        "PointsPerSecond": 56000,
                        "RotationFrequency": 10,
                        "UpperFovLimit": 10,
                        "LowerFovLimit": -30,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== RaycastLidar Test ===")
    print("Output: flat array, reshape to (-1, 5): [x, y, z, intensity, ring]")
    print("Config: 32 channels, 50 m range, 56000 pts/s, 10 Hz rotation.\n")

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
                lidar = state.get("RaycastLidar", None)
                if lidar is not None:
                    arr = np.ravel(lidar)
                    if len(arr) >= 5 and len(arr) % 5 == 0:
                        pts = arr.reshape(-1, 5)
                        n_pts = pts.shape[0]
                        xyz = pts[:, :3]
                        ranges = np.linalg.norm(xyz, axis=1)
                        mean_r = float(ranges.mean()) if n_pts > 0 else 0.0
                        print(f"Step {step:4d} | points={n_pts:5d}  mean_range={mean_r:.3f} m")
                    else:
                        print(f"Step {step:4d} | RaycastLidar raw len={len(arr)}")
                else:
                    print(f"Step {step:4d} | RaycastLidar: no data")
