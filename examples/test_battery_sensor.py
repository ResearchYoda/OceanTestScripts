"""
Test script for BatterySensor.

Agent: HoveringAUV (8 thrusters, control_scheme=0, action=np.zeros(8))
Sensor: BatterySensor
Output: [battery_%] — current battery charge level as a percentage (0–100).

Config:
  InitialLevel=100.0 (start fully charged)
  DrainRate=5.0 (%/sec at full throttle)

At zero throttle the battery drains slowly (passive drain).
Prints current level with a simple ASCII bar every 50 steps.
"""

import uuid
import numpy as np
import holoocean
import holoocean.packagemanager
import holoocean.environments

SCENARIO = {
    "name": "test_battery_sensor",
    "world": "OpenWater",
    "package_name": "Ocean",
    "main_agent": "auv0",
    "ticks_per_sec": 30,
    "agents": [
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "BatterySensor",
                    "sensor_name": "BatterySensor",
                    "configuration": {
                        "InitialLevel": 100.0,
                        "DrainRate": 5.0,
                    },
                }
            ],
            "control_scheme": 0,
            "location": [0, 0, -270],
        }
    ],
}

BAR_WIDTH = 30

def ascii_bar(level: float) -> str:
    """Return a simple ASCII progress bar for a level in [0, 100]."""
    filled = int(round(level / 100.0 * BAR_WIDTH))
    filled = max(0, min(BAR_WIDTH, filled))
    bar = "#" * filled + "-" * (BAR_WIDTH - filled)
    return f"[{bar}] {level:6.2f}%"

if __name__ == "__main__":
    print("=== BatterySensor Test ===")
    print("Agent: HoveringAUV | Output: [battery_%] charge level (0-100).")
    print("Config: InitialLevel=100%, DrainRate=5%/sec at full throttle.\n")

    binary_path = holoocean.packagemanager.get_binary_path_for_package("Ocean")

    with holoocean.environments.HoloOceanEnvironment(
        scenario=SCENARIO,
        binary_path=binary_path,
        show_viewport=False,
        verbose=False,
        uuid=str(uuid.uuid4()),
        ticks_per_sec=30,
    ) as env:
        action = np.array([0.0, 0.0, 0.0, 0.0, 30.0, 30.0, 30.0, 30.0])
        for step in range(3000):
            state = env.step(action)

            if step % 50 == 0:
                bat = state.get("BatterySensor", None)
                if bat is not None:
                    level = float(np.ravel(bat)[0])
                    print(f"Step {step:4d} | {ascii_bar(level)}")
