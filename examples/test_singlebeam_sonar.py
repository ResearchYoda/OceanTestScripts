"""
Test script for SinglebeamSonar.

Sensor: SinglebeamSonar
Output: 1D array of length RangeBins (200).
  Each element is acoustic return intensity at that range bin.
  A single beam is transmitted in a cone defined by OpeningAngle.
  The peak bin corresponds to the dominant target range.

Config:
  RangeMin=0.5 m, RangeMax=50 m, RangeBins=200
  OpeningAngle=30 deg, InitOctreeRange=60 m
Socket: SonarSocket

Display: A-scan line plot on a 600x300 canvas.
  - Normalised intensity drawn as a green line.
  - Peak range annotated.
  - Range ticks every 5 m along x-axis.
Window: "SinglebeamSonar". Press 'q' to quit.

Agent spawned at z=-275.
"""

import uuid
import numpy as np
import cv2
import holoocean
import holoocean.packagemanager
import holoocean.environments

RANGE_MIN = 0.5
RANGE_MAX = 50.0
RANGE_BINS = 200
CANVAS_W = 600
CANVAS_H = 300
PAD_LEFT = 40
PAD_RIGHT = 10
PAD_TOP = 10
PAD_BOT = 30
PLOT_W = CANVAS_W - PAD_LEFT - PAD_RIGHT
PLOT_H = CANVAS_H - PAD_TOP - PAD_BOT

SCENARIO = {
    "name": "test_singlebeam_sonar",
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
                    "sensor_type": "SinglebeamSonar",
                    "sensor_name": "SinglebeamSonar",
                    "socket": "SonarSocket",
                    "configuration": {
                        "RangeMin": RANGE_MIN,
                        "RangeMax": RANGE_MAX,
                        "RangeBins": RANGE_BINS,
                        "OpeningAngle": 30,
                        "InitOctreeRange": 60,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

def render_ascan(arr: np.ndarray) -> np.ndarray:
    """Render a normalised A-scan line plot on a dark canvas."""
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # Normalise
    a = arr.copy().astype(np.float32)
    amax = a.max()
    if amax > 1e-9:
        a /= amax

    # Build polyline points
    xs = (np.arange(RANGE_BINS) / (RANGE_BINS - 1) * PLOT_W + PAD_LEFT).astype(int)
    ys = (PAD_TOP + PLOT_H - a * PLOT_H).astype(int)
    ys = np.clip(ys, PAD_TOP, PAD_TOP + PLOT_H)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, (0, 200, 0), 1, cv2.LINE_AA)

    # Peak annotation
    peak_bin = int(np.argmax(a))
    peak_range = RANGE_MIN + peak_bin / (RANGE_BINS - 1) * (RANGE_MAX - RANGE_MIN)
    px = xs[peak_bin]
    py = ys[peak_bin]
    cv2.circle(canvas, (px, py), 4, (0, 255, 255), -1)
    cv2.putText(canvas, f"peak: {peak_range:.1f}m", (px + 5, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # Range ticks every 5 m
    tick_ranges = np.arange(0, RANGE_MAX + 1e-6, 5.0)
    for r in tick_ranges:
        tx = int((r - RANGE_MIN) / (RANGE_MAX - RANGE_MIN) * PLOT_W + PAD_LEFT)
        ty_top = PAD_TOP + PLOT_H
        cv2.line(canvas, (tx, ty_top), (tx, ty_top + 5), (150, 150, 150), 1)
        cv2.putText(canvas, f"{r:.0f}", (tx - 8, ty_top + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)

    # Axes
    cv2.line(canvas, (PAD_LEFT, PAD_TOP), (PAD_LEFT, PAD_TOP + PLOT_H), (100, 100, 100), 1)
    cv2.line(canvas, (PAD_LEFT, PAD_TOP + PLOT_H),
             (PAD_LEFT + PLOT_W, PAD_TOP + PLOT_H), (100, 100, 100), 1)

    cv2.putText(canvas, "Range (m)", (CANVAS_W // 2 - 30, CANVAS_H - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
    return canvas

if __name__ == "__main__":
    print("=== SinglebeamSonar Test ===")
    print(f"Output: {RANGE_BINS}-bin A-scan intensity array.")
    print("Displaying A-scan line plot. Press 'q' to quit.\n")

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

            sonar = state.get("SinglebeamSonar", None)
            if sonar is not None:
                arr = np.ravel(sonar).astype(np.float32)
                if len(arr) < RANGE_BINS:
                    arr = np.pad(arr, (0, RANGE_BINS - len(arr)))
                else:
                    arr = arr[:RANGE_BINS]

                canvas = render_ascan(arr)
                cv2.imshow("SinglebeamSonar", canvas)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit key pressed.")
                break

            if step % 50 == 0:
                if sonar is not None:
                    arr2 = np.ravel(sonar)
                    peak_bin = int(np.argmax(arr2))
                    peak_r = RANGE_MIN + peak_bin / (RANGE_BINS - 1) * (RANGE_MAX - RANGE_MIN)
                    print(f"Step {step:4d} | SinglebeamSonar peak at {peak_r:.2f} m (bin {peak_bin})")

    cv2.destroyAllWindows()
