"""
Test script for DepthCamera.

Sensor: DepthCamera
Output: dict with keys:
  "image" — (256, 256, 4) uint8 RGBA colour image
  "depth" — (256, 256) float32 depth map in meters

Displays the depth map with COLORMAP_JET, clipped at 20 m.
Overlays the depth value at the image centre.
Window title: "DepthCamera". Press 'q' to quit.
"""

import uuid
import numpy as np
import cv2
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_depth_camera",
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
                    "sensor_type": "DepthCamera",
                    "sensor_name": "DepthCamera",
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

CLIP_DEPTH = 20.0  # meters — depths beyond this are mapped to max colour

if __name__ == "__main__":
    print("=== DepthCamera Test ===")
    print("Output: dict with 'image' (RGBA) and 'depth' (float32, meters).")
    print(f"Displaying depth map clipped at {CLIP_DEPTH} m. Press 'q' to quit.\n")

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

            cam_data = state.get("DepthCamera", None)
            if cam_data is not None:
                # Support both dict and raw array returns
                if isinstance(cam_data, dict):
                    depth = cam_data.get("depth", None)
                else:
                    # Fallback: treat as raw depth array
                    depth = cam_data if cam_data.ndim == 2 else cam_data[:, :, 0]

                if depth is not None:
                    depth_np = np.array(depth, dtype=np.float32)
                    clipped = np.clip(depth_np, 0.0, CLIP_DEPTH)
                    normalised = (clipped / CLIP_DEPTH * 255).astype(np.uint8)
                    coloured = cv2.applyColorMap(normalised, cv2.COLORMAP_JET)

                    # Overlay centre depth value
                    h, w = depth_np.shape[:2]
                    centre_val = depth_np[h // 2, w // 2]
                    label = f"centre: {centre_val:.2f} m"
                    cv2.putText(
                        coloured, label, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                    )
                    cv2.imshow("DepthCamera", coloured)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit key pressed.")
                break

            if step % 50 == 0:
                depth_str = "N/A"
                if cam_data is not None and isinstance(cam_data, dict):
                    d = cam_data.get("depth", None)
                    if d is not None:
                        depth_str = f"{np.mean(d):.3f} m (mean)"
                print(f"Step {step:4d} | DepthCamera mean depth: {depth_str}")

    cv2.destroyAllWindows()
