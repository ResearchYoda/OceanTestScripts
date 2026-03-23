"""
Test script for SemanticSegmentationCamera.

Sensor: SemanticSegmentationCamera
Output: (256, 256, 4) uint8 RGBA image where each colour encodes a semantic class.
  Different objects/materials are rendered in distinct colours.

Displays the segmentation image live. Press 'q' to quit.
Window title: "SemanticCamera".
"""

import uuid
import numpy as np
import cv2
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_semantic_camera",
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
                    "sensor_type": "SemanticSegmentationCamera",
                    "sensor_name": "SemanticSegmentationCamera",
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
    print("=== SemanticSegmentationCamera Test ===")
    print("Output: (256, 256, 4) RGBA uint8 — colour-coded semantic labels.")
    print("Displaying live feed. Press 'q' to quit.\n")

    binary_path = holoocean.packagemanager.get_binary_path_for_package("Ocean")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=SCENARIO,
        binary_path=binary_path,
        show_viewport=True,
        verbose=False,
        uuid=str(uuid.uuid4()),
        ticks_per_sec=30,
    ) as env:
        action = np.array([0.0, 0.0, 0.0, 0.0, 60.0])
        for step in range(3000):
            state = env.step(action)

            seg = state.get("SemanticSegmentationCamera", None)
            if seg is not None:
                bgr = cv2.cvtColor(seg[:, :, :3], cv2.COLOR_RGB2BGR)
                cv2.imshow("SemanticCamera", bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit key pressed.")
                break

            if step % 50 == 0:
                shape_str = str(seg.shape) if seg is not None else "None"
                print(f"Step {step:4d} | SemanticSegmentationCamera frame shape={shape_str}")

    cv2.destroyAllWindows()
