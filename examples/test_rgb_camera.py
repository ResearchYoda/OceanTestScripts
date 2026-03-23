"""
Test script for RGBCamera.

Sensor: RGBCamera
Output: (256, 256, 4) uint8 RGBA image.
  Channels: R, G, B, A (alpha channel typically 255).

Displays the camera feed live in a cv2 window titled "RGBCamera".
Press 'q' to quit early.
"""

import uuid
import numpy as np
import cv2
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_rgb_camera",
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
                    "sensor_type": "RGBCamera",
                    "sensor_name": "RGBCamera",
                    "configuration": {
                        "CaptureWidth": 256,
                        "CaptureHeight": 256,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

if __name__ == "__main__":
    print("=== RGBCamera Test ===")
    print("Output: (256, 256, 4) RGBA uint8 image.")
    print("Displaying live feed. Press 'q' in the window to quit.\n")

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

            img = state.get("RGBCamera", None)
            if img is not None:
                bgr = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2BGR)
                cv2.imshow("RGBCamera", bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit key pressed.")
                break

            if step % 50 == 0:
                print(f"Step {step:4d} | RGBCamera frame received, shape={img.shape if img is not None else 'None'}")

    cv2.destroyAllWindows()
