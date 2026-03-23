"""
Test script for IMUSensor.

Sensor: IMUSensor
Output (ReturnBias=True): 12-element array
  [0:3]  accel    - linear acceleration (m/s^2), body frame, with noise + bias
  [3:6]  angvel   - angular velocity (rad/s), body frame, with noise + bias
  [6:9]  accelbias  - current accelerometer bias estimate (m/s^2)
  [9:12] angvelbias - current gyroscope bias estimate (rad/s)

Config:
  AccelSigma=0.00277, AngVelSigma=0.00123
  AccelBiasSigma=0.00141, AngVelBiasSigma=0.00388
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_imu_sensor",
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
                    "sensor_type": "IMUSensor",
                    "sensor_name": "IMUSensor",
                    "configuration": {
                        "AccelSigma": 0.00277,
                        "AngVelSigma": 0.00123,
                        "AccelBiasSigma": 0.00141,
                        "AngVelBiasSigma": 0.00388,
                        "ReturnBias": True,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== IMUSensor Test ===")
    print("Output: [accel(3), angvel(3), accelbias(3), angvelbias(3)]")
    print("  accel   [0:3]: linear acceleration (m/s^2) with noise+bias")
    print("  angvel  [3:6]: angular velocity (rad/s) with noise+bias")
    print("  (bias arrays follow when ReturnBias=True)\n")

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
                imu = state.get("IMUSensor", None)
                if imu is not None:
                    ax, ay, az = imu[0], imu[1], imu[2]
                    gx, gy, gz = imu[3], imu[4], imu[5]
                    print(
                        f"Step {step:4d} | "
                        f"accel=({ax:7.4f}, {ay:7.4f}, {az:7.4f}) m/s^2  "
                        f"angvel=({gx:7.4f}, {gy:7.4f}, {gz:7.4f}) rad/s"
                    )
