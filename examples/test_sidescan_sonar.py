"""
Test script for SidescanSonar.

Sensor: SidescanSonar
Output: 1D array of length RangeBins (2000) representing a single sonar ping.
  Each element is the acoustic return intensity at that range bin.
  The sonar ensonifies to both port and starboard simultaneously;
  the first half of bins corresponds to one side, second half to the other
  (or the full array may represent a single side — check package docs).

Config:
  RangeMin=0.5 m, RangeMax=40 m, RangeBins=2000
  Azimuth=170 deg, AddSigma=0.05, MultSigma=0.05
  InitOctreeRange=50 m
Socket: SonarSocket

Display: scrolling waterfall — 400 rows x 2000 cols.
Each new ping is added at the bottom; old rows scroll up.
COLORMAP_BONE applied. Range tick labels every 5 m along x-axis.
Press 'q' to quit.

Agent spawned at z=-275 (deep water above seafloor).
"""

import uuid
import numpy as np
import cv2
import holoocean
import holoocean.packagemanager
import holoocean.environments

RANGE_MIN      = 0.5
RANGE_MAX      = 30.0    # close range → strong seabed returns from Z=-270
RANGE_BINS     = 800     # 1 bin = 1 pixel wide; no resize needed
WATERFALL_ROWS = 200     # display height
IMG_W          = RANGE_BINS
IMG_H          = WATERFALL_ROWS

SCENARIO = {
    "name": "test_sidescan_sonar",
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
                    "sensor_type": "SidescanSonar",
                    "sensor_name": "SidescanSonar",
                    "socket": "SonarSocket",
                    "configuration": {
                        "RangeMin": RANGE_MIN,
                        "RangeMax": RANGE_MAX,
                        "RangeBins": RANGE_BINS,
                        "Azimuth": 170,
                        "Elevation": 0.25,
                        "AddSigma": 0.01,      # low noise → cleaner object returns
                        "MultSigma": 0.01,
                        "InitOctreeRange": 50,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

def draw_range_ticks(canvas: np.ndarray, range_min: float, range_max: float,
                     num_bins: int, tick_interval: float = 5.0) -> np.ndarray:
    """Draw range tick marks and labels along the top of the canvas."""
    out = canvas.copy()
    h = out.shape[0]
    ranges = np.arange(np.ceil(range_min / tick_interval) * tick_interval,
                       range_max + 1e-6, tick_interval)
    for r in ranges:
        x = int((r - range_min) / (range_max - range_min) * (num_bins - 1))
        if 0 <= x < num_bins:
            cv2.line(out, (x, 0), (x, 10), (200, 200, 200), 1)
            cv2.putText(out, f"{r:.0f}m", (x + 2, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
    return out

if __name__ == "__main__":
    print("=== SidescanSonar Test ===")
    print(f"Output: {RANGE_BINS}-bin ping. Displaying waterfall ({IMG_H}x{IMG_W}).")
    print("Press 'q' to quit.\n")

    binary_path = holoocean.packagemanager.get_binary_path_for_package("Ocean")

    # Waterfall buffer: rows x cols, float32 accumulator
    waterfall = np.zeros((WATERFALL_ROWS, RANGE_BINS), dtype=np.float32)

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

            ping = state.get("SidescanSonar", None)
            if ping is not None:
                arr = np.ravel(ping).astype(np.float32)
                # Pad or trim to RANGE_BINS
                if len(arr) < RANGE_BINS:
                    arr = np.pad(arr, (0, RANGE_BINS - len(arr)))
                else:
                    arr = arr[:RANGE_BINS]

                # Scroll waterfall up and add new ping at bottom
                waterfall[:-1] = waterfall[1:]
                waterfall[-1] = arr

                # Percentile normalisation for better object contrast
                wf_norm = waterfall.copy()
                lo = np.percentile(wf_norm, 2)
                hi = np.percentile(wf_norm, 98)
                if hi > lo:
                    wf_norm = np.clip((wf_norm - lo) / (hi - lo), 0.0, 1.0)
                wf_u8 = (wf_norm * 255).astype(np.uint8)
                coloured = cv2.applyColorMap(wf_u8, cv2.COLORMAP_HOT)

                # Range ticks
                coloured = draw_range_ticks(coloured, RANGE_MIN, RANGE_MAX,
                                            RANGE_BINS, tick_interval=5.0)

                cv2.imshow("SidescanSonar Waterfall", coloured)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Quit key pressed.")
                break

            if step % 50 == 0:
                peak = float(np.ravel(ping).max()) if ping is not None else 0.0
                print(f"Step {step:4d} | SidescanSonar peak intensity = {peak:.4f}")

    cv2.destroyAllWindows()
