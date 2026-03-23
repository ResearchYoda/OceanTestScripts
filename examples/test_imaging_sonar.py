"""
Test script for ImagingSonar.

Sensor: ImagingSonar
Output: 2D array of shape (AzimuthBins, RangeBins) = (512, 512).
  Each element is the acoustic return intensity for that azimuth/range cell.

Config:
  RangeBins=512, AzimuthBins=512
  RangeMin=1 m, RangeMax=40 m
  Elevation=20 deg, Azimuth=120 deg
  InitOctreeRange=50
  AddSigma=0.15, MultSigma=0.2, RangeSigma=0.1
  AzimuthStreaks=-1, ScaleNoise=True, MultiPath=True
Socket: SonarSocket

Display: polar fan image (IMG_W=600, IMG_H=500).
Polar-to-pixel lookup is precomputed once outside the loop using numpy vectorised ops.
COLORMAP_HOT applied. Range arc labels every 5 m.
Press 'q' to quit.

Agent spawned at z=-275.
"""

import uuid
import numpy as np
import cv2
import holoocean
import holoocean.packagemanager
import holoocean.environments

RANGE_MIN = 1.0
RANGE_MAX = 40.0
RANGE_BINS = 512
AZ_BINS = 512
AZ_DEG = 120.0
IMG_W = 600
IMG_H = 500

SCENARIO = {
    "name": "test_imaging_sonar",
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
                    "sensor_type": "ImagingSonar",
                    "sensor_name": "ImagingSonar",
                    "socket": "SonarSocket",
                    "configuration": {
                        "RangeBins": RANGE_BINS,
                        "AzimuthBins": AZ_BINS,
                        "RangeMin": RANGE_MIN,
                        "RangeMax": RANGE_MAX,
                        "Elevation": 20,
                        "Azimuth": AZ_DEG,
                        "InitOctreeRange": 50,
                        "AddSigma": 0.15,
                        "MultSigma": 0.2,
                        "RangeSigma": 0.1,
                        "AzimuthStreaks": -1,
                        "ScaleNoise": True,
                        "MultiPath": True,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

# --- Precompute polar->pixel lookup (vectorised, no Python for-loops) ---
_cx = IMG_W // 2
_cy = IMG_H - 10
yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
dx = xx - _cx
dy = _cy - yy
r_px = np.sqrt(dx * dx + dy * dy)
r_m = r_px / _cy * RANGE_MAX
angle_deg = np.degrees(np.arctan2(dx, dy))
_valid = (
    (r_m >= RANGE_MIN) & (r_m <= RANGE_MAX) & (np.abs(angle_deg) <= AZ_DEG / 2)
)
_r_idx = np.clip(
    ((r_m - RANGE_MIN) / (RANGE_MAX - RANGE_MIN) * (RANGE_BINS - 1)).astype(int),
    0, RANGE_BINS - 1,
)
_a_idx = np.clip(
    ((angle_deg + AZ_DEG / 2) / AZ_DEG * (AZ_BINS - 1)).astype(int),
    0, AZ_BINS - 1,
)

def draw_range_arcs(canvas: np.ndarray, interval: float = 5.0) -> np.ndarray:
    """Draw range arcs and labels on the polar fan image."""
    out = canvas.copy()
    ranges = np.arange(interval, RANGE_MAX + 1e-6, interval)
    for r in ranges:
        radius_px = int(r / RANGE_MAX * _cy)
        cv2.ellipse(out, (_cx, _cy), (radius_px, radius_px), 0,
                    -(90 + AZ_DEG / 2), -(90 - AZ_DEG / 2),
                    (100, 100, 100), 1)
        label_x = _cx + int(radius_px * np.sin(np.radians(-AZ_DEG / 2 + 5)))
        label_y = _cy - int(radius_px * np.cos(np.radians(-AZ_DEG / 2 + 5)))
        cv2.putText(out, f"{r:.0f}m", (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)
    return out

if __name__ == "__main__":
    print("=== ImagingSonar Test ===")
    print(f"Output: ({AZ_BINS}, {RANGE_BINS}) intensity array.")
    print(f"Displaying polar fan ({IMG_W}x{IMG_H}). Press 'q' to quit.\n")

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

            sonar = state.get("ImagingSonar", None)
            if sonar is not None:
                data = np.array(sonar, dtype=np.float32)
                # Ensure shape is (AZ_BINS, RANGE_BINS)
                if data.ndim == 1:
                    data = data.reshape(AZ_BINS, RANGE_BINS)
                elif data.shape == (RANGE_BINS, AZ_BINS):
                    data = data.T

                # Normalise
                dmax = data.max()
                if dmax > 1e-9:
                    data = data / dmax

                # Build polar image using precomputed indices
                canvas = np.zeros((IMG_H, IMG_W), dtype=np.float32)
                canvas[_valid] = data[_a_idx[_valid], _r_idx[_valid]]

                canvas_u8 = (canvas * 255).astype(np.uint8)
                coloured = cv2.applyColorMap(canvas_u8, cv2.COLORMAP_HOT)
                coloured = draw_range_arcs(coloured, interval=5.0)

                cv2.imshow("ImagingSonar", coloured)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit key pressed.")
                break

            if step % 50 == 0:
                peak = float(np.array(sonar).max()) if sonar is not None else 0.0
                print(f"Step {step:4d} | ImagingSonar peak = {peak:.4f}")

    cv2.destroyAllWindows()
