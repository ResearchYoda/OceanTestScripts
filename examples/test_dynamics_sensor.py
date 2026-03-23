"""
Test script for DynamicsSensor.

Sensor: DynamicsSensor
Output: 18-element array
  [0:3]   accel     — linear acceleration (m/s^2)
  [3:6]   vel       — linear velocity (m/s)
  [6:9]   pos       — position (m)
  [9:12]  ang_accel — angular acceleration (rad/s^2)
  [12:15] ang_vel   — angular velocity (rad/s)
  [15:18] rpy       — roll/pitch/yaw (rad when UseRPY=True)

Config: UseCOM=True (use centre of mass), UseRPY=True (return RPY instead of quaternion).
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_dynamics_sensor",
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
                    "sensor_type": "DynamicsSensor",
                    "sensor_name": "DynamicsSensor",
                    "configuration": {
                        "UseCOM": True,
                        "UseRPY": True,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== DynamicsSensor Test ===")
    print("Output: 18-element array")
    print("  [0:3]   accel (m/s^2), [3:6] vel (m/s), [6:9] pos (m)")
    print("  [9:12]  ang_accel (rad/s^2), [12:15] ang_vel (rad/s), [15:18] rpy (rad)")
    print("Printing pos and vel every 50 steps.\n")

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
                dyn = state.get("DynamicsSensor", None)
                if dyn is not None:
                    arr = np.ravel(dyn)
                    pos = arr[6:9]
                    vel = arr[3:6]
                    speed = np.linalg.norm(vel)
                    print(
                        f"Step {step:4d} | "
                        f"pos=({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}) m  "
                        f"vel=({vel[0]:7.4f}, {vel[1]:7.4f}, {vel[2]:7.4f}) m/s  "
                        f"|v|={speed:.4f}"
                    )
