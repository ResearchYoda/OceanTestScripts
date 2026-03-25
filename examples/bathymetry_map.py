"""
bathymetry_map.py — Extract a bathymetry map using ProfilingSonar in HoloOcean.

Algorithm
---------
For each sonar frame:
  1. Bottom-detect: per azimuth beam find the range bin with the highest
     intensity above a threshold.  That range is the seafloor return.
  2. Convert (azimuth_angle, range) → 3-D world coordinates using the
     sonar geometry and the vehicle pose from DynamicsSensor.
  3. Bin each seafloor point into a 2-D horizontal grid and track the
     mean depth per cell.
  4. Render the accumulated grid as a colour-coded depth image.

Coordinate conventions (HoloOcean / UE5)
-----------------------------------------
  X = forward,  Y = starboard (right),  Z = up.
  DynamicsSensor (UseRPY=True) returns an 18-element vector:
      [accel(3), vel(3), pos(3), ang_accel(3), ang_vel(3), rpy(3)]
  pos is in metres; rpy is roll, pitch, yaw in degrees.

Sonar mounting assumption
--------------------------
  Bore-sight points straight down (−Z body axis).
  The azimuth fan spreads in the body-Y (port ↔ starboard) direction.
  For a beam at azimuth angle θ and range r:
      body_y_offset = r * sin(θ)    (cross-track; positive = starboard)
      body_z_offset = −r * cos(θ)  (downward; negative in Z-up frame)
  Rotating into the world frame via heading ψ:
      Δeasting  = body_y * (−sin ψ)
      Δnorthing = body_y * ( cos ψ)
      Δdepth    = body_z          (no rotation needed for vertical)

Usage
-----
  python bathymetry_map.py [--steps N] [--save bathy.npz] [--no-display]

Controls (OpenCV window)  — always active
-----------------------------------------
  Esc          quit
  P            save the bathymetry grid   (also saved on exit if --save)
  R            reset the bathymetry accumulator
  M            toggle manual / auto (lawnmower) mode
  3            refresh the 3-D surface plot now

Manual keyboard controls  — active in manual mode only
-------------------------------------------------------
  W / ↑        forward thrust
  S / ↓        reverse thrust
  A / ←        yaw left  (turn port)
  D / →        yaw right (turn starboard)
  Z            strafe left  (port)
  X            strafe right (starboard)
  Space        ascend  (+Z thrust)
  C            descend (−Z thrust)

  Requires pynput for held-key detection.  Install with:
      pip install pynput
  If pynput is unavailable the script falls back to single-tap key events
  via OpenCV (keys must be tapped repeatedly rather than held).
"""

import argparse
import uuid
import threading
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")          # non-blocking backend; swap to "Qt5Agg" if needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers the 3-D projection
import holoocean
import holoocean.packagemanager
import holoocean.environments

try:
    from pynput import keyboard as _pynput_kb
    _PYNPUT_AVAILABLE = True
except ImportError:
    _PYNPUT_AVAILABLE = False

# ── Sonar configuration ────────────────────────────────────────────────────────
RANGE_MIN  = 1.0    # metres
RANGE_MAX  = 60.0   # metres
RANGE_BINS = 512
AZ_BINS    = 512
AZ_DEG     = 120.0  # total azimuth swath in degrees

# Pre-computed lookup arrays (constant for a given sensor config)
_AZ_ANGLES = np.linspace(-AZ_DEG / 2, AZ_DEG / 2, AZ_BINS)   # (AZ_BINS,) deg
_RANGES     = np.linspace(RANGE_MIN, RANGE_MAX, RANGE_BINS)    # (RANGE_BINS,) m

# ── Bottom-detection parameters ────────────────────────────────────────────────
# Only accept a beam's peak if it is at least this fraction of the frame max.
INTENSITY_THRESHOLD = 0.05

# ── Bathymetry grid ────────────────────────────────────────────────────────────
GRID_RES   = 0.5    # metres per cell
GRID_HALF  = 100.0  # grid spans ±GRID_HALF metres from the world origin
GRID_N     = int(2 * GRID_HALF / GRID_RES)   # cells per axis (400 × 400)

# ── Sonar fan display (polar image shown alongside the map) ───────────────────
FAN_W, FAN_H = 500, 450
_fcx = FAN_W // 2
_fcy = FAN_H - 20   # sonar origin at bottom-centre of the image

# Build polar → pixel lookup once (vectorised)
_fyy, _fxx   = np.mgrid[0:FAN_H, 0:FAN_W]
_fdx         = _fxx - _fcx
_fdy         = _fcy - _fyy
_fr_px       = np.hypot(_fdx, _fdy)
_fr_m        = _fr_px / _fcy * RANGE_MAX
_fangle      = np.degrees(np.arctan2(_fdx, _fdy))
_fvalid      = (_fr_m >= RANGE_MIN) & (_fr_m <= RANGE_MAX) & (np.abs(_fangle) <= AZ_DEG / 2)
_fr_idx      = np.clip(((_fr_m - RANGE_MIN) / (RANGE_MAX - RANGE_MIN) * (RANGE_BINS - 1)).astype(int),
                       0, RANGE_BINS - 1)
_faz_idx     = np.clip(((_fangle + AZ_DEG / 2) / AZ_DEG * (AZ_BINS - 1)).astype(int),
                       0, AZ_BINS - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────────────────────

class BathymetryGrid:
    """Accumulates sonar bottom-detections into a 2-D depth grid."""

    def __init__(self):
        self.depth_sum   = np.zeros((GRID_N, GRID_N), dtype=np.float64)
        self.depth_count = np.zeros((GRID_N, GRID_N), dtype=np.int32)
        self.n_points    = 0

    def reset(self):
        self.depth_sum[:]   = 0.0
        self.depth_count[:] = 0
        self.n_points       = 0

    def _world_to_cell(self, wx, wy):
        col = ((wx + GRID_HALF) / GRID_RES).astype(int)
        row = ((wy + GRID_HALF) / GRID_RES).astype(int)
        return row, col

    def add_points(self, easting, northing, depth):
        """Bin an array of world-frame (easting, northing, depth) points."""
        row, col = self._world_to_cell(easting, northing)
        mask = (row >= 0) & (row < GRID_N) & (col >= 0) & (col < GRID_N)
        r, c, d = row[mask], col[mask], depth[mask]
        np.add.at(self.depth_sum,   (r, c), d)
        np.add.at(self.depth_count, (r, c), 1)
        self.n_points += mask.sum()

    @property
    def mean_depth(self):
        """2-D array of mean depth (NaN where no data)."""
        with np.errstate(invalid="ignore"):
            return np.where(self.depth_count > 0,
                            self.depth_sum / self.depth_count,
                            np.nan)

    def save(self, path):
        np.savez(path,
                 depth_sum=self.depth_sum,
                 depth_count=self.depth_count,
                 grid_res=np.float64(GRID_RES),
                 grid_half=np.float64(GRID_HALF))
        print(f"[bathy] Saved grid to {path}  ({self.n_points} points total)")


def detect_bottom(sonar_frame):
    """
    Bottom-detect a single sonar frame.

    Parameters
    ----------
    sonar_frame : ndarray, shape (RANGE_BINS, AZ_BINS)

    Returns
    -------
    az_angles : ndarray (M,) — azimuth angles of valid beams, degrees
    ranges    : ndarray (M,) — detected seafloor range per beam, metres
    """
    frame_max = sonar_frame.max()
    if frame_max < 1e-9:
        return np.empty(0), np.empty(0)

    # Index of peak intensity along range axis for every azimuth beam
    peak_idx = np.argmax(sonar_frame, axis=0)                        # (AZ_BINS,)
    peak_val = sonar_frame[peak_idx, np.arange(AZ_BINS)]             # (AZ_BINS,)

    valid = peak_val >= INTENSITY_THRESHOLD * frame_max
    return _AZ_ANGLES[valid], _RANGES[peak_idx[valid]]


def sonar_to_world(az_deg, ranges, pos, yaw_deg):
    """
    Convert (azimuth, range) detections to world-frame (easting, northing, depth).

    Parameters
    ----------
    az_deg  : ndarray (M,) — azimuth angles in degrees
    ranges  : ndarray (M,) — seafloor range in metres
    pos     : array (3,) — vehicle position [x, y, z] in metres
    yaw_deg : float — vehicle yaw (heading) in degrees

    Returns
    -------
    easting, northing, depth : ndarrays (M,) in metres
    """
    az_rad  = np.radians(az_deg)
    yaw_rad = np.radians(yaw_deg)

    # Body-frame offsets (bore-sight = −Z, fan = Y)
    body_y = ranges * np.sin(az_rad)    # cross-track (starboard positive)
    body_z = -ranges * np.cos(az_rad)   # downward (negative in Z-up)

    # Rotate cross-track into world frame via heading
    easting  = pos[0] + body_y * (-np.sin(yaw_rad))
    northing = pos[1] + body_y * ( np.cos(yaw_rad))
    depth    = pos[2] + body_z           # absolute Z of seafloor in world

    return easting, northing, depth


def render_fan(sonar_frame):
    """Render a sonar frame as a colour polar fan image."""
    canvas = np.zeros((FAN_H, FAN_W), dtype=np.float32)
    canvas[_fvalid] = sonar_frame[_fr_idx[_fvalid], _faz_idx[_fvalid]]
    c_max = canvas.max()
    canvas_u8 = (canvas / c_max * 255).astype(np.uint8) if c_max > 0 \
                else canvas.astype(np.uint8)
    fan = cv2.applyColorMap(canvas_u8, cv2.COLORMAP_HOT)
    fan[~_fvalid] = (15, 15, 15)

    # Range arc overlays
    for r_m in range(10, int(RANGE_MAX) + 1, 10):
        r_px = int(r_m / RANGE_MAX * _fcy)
        cv2.ellipse(fan, (_fcx, _fcy), (r_px, r_px), 0,
                    -90 - AZ_DEG / 2, -90 + AZ_DEG / 2,
                    (60, 60, 60), 1)
        cv2.putText(fan, f"{r_m}m", (_fcx + 4, _fcy - r_px + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)
    return fan


def render_bathy(grid, vehicle_pos=None):
    """
    Render the bathymetry grid as a colour depth image.

    Shallow areas → bright (COLORMAP_OCEAN blue); deep → dark.
    The vehicle's current cell is marked with a white dot.
    """
    dm = grid.mean_depth
    valid = ~np.isnan(dm)

    img = np.zeros((GRID_N, GRID_N, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)   # unvisited = dark grey

    if valid.any():
        d_min = dm[valid].min()
        d_max = dm[valid].max()
        d_rng = d_max - d_min if d_max != d_min else 1.0

        norm = np.zeros((GRID_N, GRID_N), dtype=np.float32)
        norm[valid] = (dm[valid] - d_min) / d_rng   # 0 = shallowest
        norm_u8 = (norm * 255).astype(np.uint8)
        coloured = cv2.applyColorMap(norm_u8, cv2.COLORMAP_OCEAN)
        img[valid] = coloured[valid]

        # Depth scale bar (left edge)
        bar_h = min(GRID_N, 200)
        for i in range(bar_h):
            v = int(i / bar_h * 255)
            colour = cv2.applyColorMap(np.array([[v]], dtype=np.uint8), cv2.COLORMAP_OCEAN)[0, 0]
            img[GRID_N - bar_h + i, 0:8] = colour
        cv2.putText(img, f"{d_min:.1f}m", (10, GRID_N - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(img, f"{d_max:.1f}m", (10, GRID_N - bar_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

    # Mark vehicle position
    if vehicle_pos is not None:
        vcol = int((vehicle_pos[0] + GRID_HALF) / GRID_RES)
        vrow = int((vehicle_pos[1] + GRID_HALF) / GRID_RES)
        if 0 <= vrow < GRID_N and 0 <= vcol < GRID_N:
            cv2.circle(img, (vcol, vrow), 4, (255, 255, 255), -1)
            cv2.circle(img, (vcol, vrow), 5, (0, 0, 0), 1)

    # Scale to a sensible display size
    display_px = 600
    scale = display_px / GRID_N
    if scale != 1.0:
        img = cv2.resize(img, (display_px, display_px), interpolation=cv2.INTER_NEAREST)

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Scenario
# ─────────────────────────────────────────────────────────────────────────────

SCENARIO = {
    "name": "bathymetry_mapping",
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
                    # 18-element vector: accel(3) vel(3) pos(3) ang_accel(3) ang_vel(3) rpy(3)
                    "sensor_type": "DynamicsSensor",
                    "sensor_name": "DynamicsSensor",
                    "configuration": {
                        "UseCOM": True,
                        "UseRPY": True,
                    },
                },
                {
                    "sensor_type": "ProfilingSonar",
                    "sensor_name": "ProfilingSonar",
                    "socket": "SonarSocket",
                    "configuration": {
                        "RangeBins": RANGE_BINS,
                        "AzimuthBins": AZ_BINS,
                        "RangeMin": RANGE_MIN,
                        "RangeMax": RANGE_MAX,
                        "Elevation": 1,
                        "Azimuth": AZ_DEG,
                        "InitOctreeRange": 70,
                        "AddSigma": 0.05,
                        "MultSigma": 0.1,
                        "RangeSigma": 0.05,
                        "AzimuthStreaks": -1,
                        "ScaleNoise": True,
                        "MultiPath": False,
                    },
                },
            ],
            # control_scheme=0: 5-element command [fx, fy, fz, tx, ty, tz] (first 5 used)
            # or a simple forward thrust command
            "control_scheme": 0,
            "location": [0, 0, -270],  # start 2.7 m below the surface
        }
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Manual keyboard controller
# ─────────────────────────────────────────────────────────────────────────────

# Force / torque magnitudes (body frame, N and N·m)
KB_FWD_FORCE   = 300.0   # W / ↑ — forward
KB_SIDE_FORCE  = 200.0   # Z / X — strafe
KB_VERT_FORCE  = 150.0   # Space / C — heave
KB_YAW_TORQUE  = 50.0    # A / D — yaw

# OpenCV key codes (returned by waitKey)
_CV_ARROW_UP    = 82
_CV_ARROW_DOWN  = 84
_CV_ARROW_LEFT  = 81
_CV_ARROW_RIGHT = 83


class KeyboardController:
    """
    Manual AUV controller driven by keyboard input.

    Action vector  [Fx, Fy, Fz, Tx, Tz]  (body-frame forces / yaw torque):
        Fx  > 0 → forward
        Fy  > 0 → starboard (strafe right)
        Fz  > 0 → up (ascend)
        Tx    not used (always 0)
        Tz  > 0 → yaw right

    Two input backends are supported:
      • pynput  — tracks which keys are *currently held*, giving smooth control.
      • OpenCV  — single-tap events; press a key once to apply one pulse of thrust.

    Key bindings
    ------------
      W / ↑     forward      S / ↓   reverse
      A / ←     yaw left     D / →   yaw right
      Z         strafe left  X       strafe right
      Space     ascend       C       descend
    """

    def __init__(self):
        self._lock      = threading.Lock()
        self._held      = set()          # pynput: currently pressed key names
        self._cv_pulse  = set()          # OpenCV fallback: one-shot key names

        if _PYNPUT_AVAILABLE:
            self._listener = _pynput_kb.Listener(
                on_press=self._on_press, on_release=self._on_release)
            self._listener.start()
        else:
            self._listener = None
            print("[keyboard] pynput not found — using OpenCV tap mode "
                  "(tap keys repeatedly for continuous thrust).")

    # ── pynput callbacks ──────────────────────────────────────────────────────

    @staticmethod
    def _key_name(key):
        """Normalise a pynput Key or KeyCode to a lowercase string."""
        try:
            return key.char.lower()           # printable key  → 'w', 'a', …
        except AttributeError:
            return str(key)                   # special key    → 'Key.space', …

    def _on_press(self, key):
        with self._lock:
            self._held.add(self._key_name(key))

    def _on_release(self, key):
        with self._lock:
            self._held.discard(self._key_name(key))

    # ── OpenCV fallback ───────────────────────────────────────────────────────

    def feed_cv_key(self, cv_key):
        """
        Call once per frame with the return value of cv2.waitKey().
        Maps the code to a logical key name and queues a one-shot pulse.
        """
        mapping = {
            ord('w'): 'w',  ord('s'): 's',
            ord('a'): 'a',  ord('d'): 'd',
            ord('z'): 'z',  ord('x'): 'x',
            ord(' '): 'Key.space',
            ord('c'): 'c',
            _CV_ARROW_UP:    'w',
            _CV_ARROW_DOWN:  's',
            _CV_ARROW_LEFT:  'a',
            _CV_ARROW_RIGHT: 'd',
        }
        name = mapping.get(cv_key & 0xFF)
        if name:
            with self._lock:
                self._cv_pulse.add(name)

    # ── Action generation ─────────────────────────────────────────────────────

    def get_action(self):
        """Return the current 5-element force/torque command."""
        with self._lock:
            if _PYNPUT_AVAILABLE:
                active = set(self._held)
            else:
                active = set(self._cv_pulse)
                self._cv_pulse.clear()   # one-shot: consume after reading

        fx = fy = fz = tz = 0.0

        if 'w' in active or 'Key.up' in active:
            fx += KB_FWD_FORCE
        if 's' in active or 'Key.down' in active:
            fx -= KB_FWD_FORCE
        if 'a' in active or 'Key.left' in active:
            tz -= KB_YAW_TORQUE
        if 'd' in active or 'Key.right' in active:
            tz += KB_YAW_TORQUE
        if 'z' in active:
            fy -= KB_SIDE_FORCE
        if 'x' in active:
            fy += KB_SIDE_FORCE
        if 'Key.space' in active:
            fz += KB_VERT_FORCE
        if 'c' in active:
            fz -= KB_VERT_FORCE

        return np.array([fx, fy, fz, 0.0, tz], dtype=np.float32)

    def stop(self):
        if self._listener is not None:
            self._listener.stop()


def _draw_kb_hud(img):
    """Overlay the keyboard-control legend on the bathymetry map image."""
    lines = [
        "-- MANUAL MODE --",
        "W/S  fwd/rev     A/D  yaw L/R",
        "Z/X  strafe L/R  Spc/C  up/down",
        "M  auto   P  save   R  reset   3  3D",
    ]
    x0, y0, dy = 8, img.shape[0] - 60, 14
    for i, line in enumerate(lines):
        colour = (0, 255, 180) if i == 0 else (180, 220, 180)
        cv2.putText(img, line, (x0, y0 + i * dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Lawnmower survey path generator
# ─────────────────────────────────────────────────────────────────────────────

class LawnmowerController:
    """
    Simple open-loop lawnmower survey in the X–Y plane.

    The TorpedoAUV is driven by a 5-element force/torque command:
        [Fx, Fy, Fz, Tx, Tz]   (body frame)
    We apply forward thrust (Fx) and yaw torque (Tz) to turn at the end of
    each leg.  Z thrust is zero because the AUV naturally holds depth.
    """

    def __init__(self, leg_length=40.0, n_legs=5, swath=20.0,
                 forward_force=200.0, yaw_torque=30.0):
        self.leg_length     = leg_length
        self.n_legs         = n_legs
        self.swath          = swath   # cross-track spacing between legs
        self.forward_force  = forward_force
        self.yaw_torque     = yaw_torque

        # State machine
        self._leg         = 0
        self._turning     = False
        self._turn_steps  = 0
        self._TURN_STEPS  = 60   # ticks to complete a 180° turn at 30 Hz
        self._done        = False

    @property
    def done(self):
        return self._done

    def step(self, pos, yaw_deg):
        """Return a 5-element action array for the current survey state."""
        if self._done:
            return np.zeros(5)

        if self._turning:
            self._turn_steps += 1
            direction = 1 if self._leg % 2 == 0 else -1
            action = np.array([self.forward_force * 0.3, 0.0, 0.0, 0.0,
                               direction * self.yaw_torque])
            if self._turn_steps >= self._TURN_STEPS:
                self._turning    = False
                self._turn_steps = 0
                self._leg       += 1
                if self._leg >= self.n_legs:
                    self._done = True
            return action

        # Straight-line leg
        action = np.array([self.forward_force, 0.0, 0.0, 0.0, 0.0])

        # Check if we have travelled far enough on this leg (distance from origin)
        leg_dist = abs(pos[0] if self._leg % 2 == 0 else pos[1])
        if leg_dist >= self.leg_length / 2 * (self._leg + 1) / self.n_legs * self.n_legs:
            self._turning    = True
            self._turn_steps = 0

        return action


# ─────────────────────────────────────────────────────────────────────────────
# 3-D surface map
# ─────────────────────────────────────────────────────────────────────────────

# Downsample factor: render every Nth cell so the surface stays interactive
_3D_DOWNSAMPLE = 4   # 400×400 → 100×100 points

class Map3D:
    """
    Live 3-D surface plot of the bathymetry grid.

    Depth is displayed as elevation (Z-up), so the seafloor topology is
    immediately visible: hills appear as bumps, trenches as holes.
    Call update() to refresh the plot; the window is non-blocking.
    """

    def __init__(self):
        plt.ion()
        self._fig = plt.figure("Bathymetry 3D", figsize=(8, 6))
        self._ax  = self._fig.add_subplot(111, projection="3d")
        self._fig.tight_layout()
        self._surf = None

        # Axis labels
        self._ax.set_xlabel("Easting (m)")
        self._ax.set_ylabel("Northing (m)")
        self._ax.set_zlabel("Depth (m)")
        self._ax.set_title("Seafloor Bathymetry")

        # Pre-build the X/Y coordinate grids at the downsampled resolution
        step     = _3D_DOWNSAMPLE
        coords   = np.arange(0, GRID_N, step)
        world    = (coords * GRID_RES) - GRID_HALF   # cell index → metres
        self._X, self._Y = np.meshgrid(world, world)  # both (n, n)
        self._n  = len(coords)

        plt.pause(0.001)

    def update(self, grid):
        """Redraw the surface from the current BathymetryGrid."""
        dm = grid.mean_depth   # (GRID_N, GRID_N), NaN where unvisited

        # Downsample
        step = _3D_DOWNSAMPLE
        Z = dm[::step, ::step]   # (n, n)

        # Replace NaN with the shallowest known depth so the surface stays
        # connected; NaN cells will appear at the "ceiling".
        valid = ~np.isnan(Z)
        if not valid.any():
            plt.pause(0.001)
            return
        fill = Z[valid].max()
        Z_filled = np.where(valid, Z, fill)

        # Remove old surface
        if self._surf is not None:
            self._surf.remove()

        self._surf = self._ax.plot_surface(
            self._X, self._Y, Z_filled,
            facecolors=self._depth_to_colour(Z_filled, Z[valid].min(), Z[valid].max()),
            linewidth=0, antialiased=False, shade=True,
        )

        # Keep Z axis tight around real data
        self._ax.set_zlim(Z[valid].min() - 1, Z[valid].max() + 1)

        self._fig.canvas.draw_idle()
        plt.pause(0.001)

    @staticmethod
    def _depth_to_colour(Z, z_min, z_max):
        """Map depth values to RGBA using the 'terrain' colormap."""
        cmap  = plt.get_cmap("terrain")
        z_rng = z_max - z_min if z_max != z_min else 1.0
        norm  = (Z - z_min) / z_rng          # 0 = deepest, 1 = shallowest
        # Invert so shallow = bright (terrain convention)
        return cmap(1.0 - norm)

    def close(self):
        plt.close(self._fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="HoloOcean bathymetry map extractor")
    parser.add_argument("--steps",      type=int,   default=5000,
                        help="Total simulation steps (default: 5000)")
    parser.add_argument("--save",       type=str,   default="bathy.npz",
                        help="Output file for the bathymetry grid (default: bathy.npz)")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable OpenCV windows (headless mode)")
    parser.add_argument("--manual",     action="store_true",
                        help="Start in manual keyboard control mode")
    parser.add_argument("--no-3d",      action="store_true",
                        help="Disable the 3-D surface map window")
    parser.add_argument("--3d-interval", type=int, default=300, dest="interval_3d",
                        help="Refresh 3-D map every N steps (default: 300)")
    args = parser.parse_args()

    show        = not args.no_display
    show_3d     = show and not args.no_3d
    manual_mode = args.manual

    grid        = BathymetryGrid()
    auto_ctrl   = LawnmowerController(leg_length=80.0, n_legs=6, swath=15.0,
                                      forward_force=300.0, yaw_torque=40.0)
    kb_ctrl     = KeyboardController()
    map3d       = Map3D() if show_3d else None

    binary_path = holoocean.packagemanager.get_binary_path_for_package("Ocean")

    print("=== HoloOcean Bathymetry Extractor ===")
    print(f"  Grid:        {GRID_N}×{GRID_N} cells, {GRID_RES} m/cell "
          f"(covers ±{GRID_HALF} m)")
    print(f"  Sonar:       {AZ_BINS}az × {RANGE_BINS}rng bins, "
          f"{RANGE_MIN}–{RANGE_MAX} m, ±{AZ_DEG/2}° azimuth")
    print(f"  Steps:       {args.steps}")
    print(f"  Output:      {args.save}")
    print(f"  Mode:        {'MANUAL' if manual_mode else 'AUTO (lawnmower)'}")
    print(f"  3-D map:     {'every ' + str(args.interval_3d) + ' steps' if show_3d else 'off'}")
    if show:
        print("  Controls:    Esc quit | P save | R reset | M toggle | 3 refresh 3D")
        if _PYNPUT_AVAILABLE:
            print("  Keyboard:    pynput (held-key mode)")
        else:
            print("  Keyboard:    OpenCV tap mode (pynput not installed)")
    print()

    with holoocean.environments.HoloOceanEnvironment(
        scenario=SCENARIO,
        binary_path=binary_path,
        show_viewport=True,
        verbose=False,
        uuid=str(uuid.uuid4()),
        ticks_per_sec=SCENARIO["ticks_per_sec"],
    ) as env:

        state = env.reset()
        pos   = np.zeros(3)
        yaw   = 0.0

        for step in range(args.steps):
            # ── Retrieve pose ─────────────────────────────────────────────────
            dyn = state.get("DynamicsSensor")
            if dyn is not None:
                dyn = np.asarray(dyn, dtype=np.float32)
                pos = dyn[6:9]    # [x, y, z] in metres
                yaw = float(dyn[17])  # dyn[15:18] = [roll, pitch, yaw]

            # ── Control ───────────────────────────────────────────────────────
            if manual_mode:
                action = kb_ctrl.get_action()
            else:
                action = auto_ctrl.step(pos, yaw)

            state = env.step(action)

            # ── Bathymetry extraction ─────────────────────────────────────────
            sonar = state.get("ProfilingSonar")
            if sonar is not None:
                sonar_frame = np.asarray(sonar, dtype=np.float32)
                # Ensure shape is (RANGE_BINS, AZ_BINS)
                if sonar_frame.ndim == 1:
                    sonar_frame = sonar_frame.reshape(RANGE_BINS, AZ_BINS)
                elif sonar_frame.shape == (AZ_BINS, RANGE_BINS):
                    sonar_frame = sonar_frame.T

                az, rng = detect_bottom(sonar_frame)
                if az.size > 0:
                    e, n, d = sonar_to_world(az, rng, pos, yaw)
                    grid.add_points(e, n, d)

                # ── Display ───────────────────────────────────────────────────
                if show:
                    fan_img   = render_fan(sonar_frame)
                    bathy_img = render_bathy(grid, vehicle_pos=pos)

                    # Top HUD
                    mode_str = "MANUAL" if manual_mode else "AUTO"
                    cv2.putText(bathy_img,
                                f"[{mode_str}] step {step}  "
                                f"pos ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) m  "
                                f"yaw {yaw:.1f}°",
                                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
                    cv2.putText(bathy_img,
                                f"points: {grid.n_points}  "
                                f"cells: {(grid.depth_count > 0).sum()}",
                                (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

                    # Bottom HUD: show key legend in manual mode
                    if manual_mode:
                        _draw_kb_hud(bathy_img)
                    else:
                        cv2.putText(bathy_img,
                                    "Esc quit | P save | R reset | M manual | 3 refresh 3D",
                                    (8, bathy_img.shape[0] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1)

                    cv2.imshow("ProfilingSonar Fan", fan_img)
                    cv2.imshow("Bathymetry Map",     bathy_img)

                    raw_key = cv2.waitKey(1)
                    key     = raw_key & 0xFF

                    # Navigation keys fed to keyboard controller (OpenCV fallback)
                    if not _PYNPUT_AVAILABLE and manual_mode:
                        kb_ctrl.feed_cv_key(raw_key)

                    # App-level keys  (no overlap with manual driving keys)
                    if key == 27:                   # Esc
                        print("Quit key pressed.")
                        break
                    elif key == ord("p"):
                        grid.save(args.save)
                    elif key == ord("r"):
                        print("[bathy] Grid reset.")
                        grid.reset()
                    elif key == ord("m"):
                        manual_mode = not manual_mode
                        label = "MANUAL" if manual_mode else "AUTO (lawnmower)"
                        print(f"[ctrl] Switched to {label} mode.")
                    elif key == ord("3") and map3d is not None:
                        map3d.update(grid)

                # ── Periodic 3-D refresh ──────────────────────────────────────
                if map3d is not None and step % args.interval_3d == 0 and step > 0:
                    map3d.update(grid)

            # ── Console log ───────────────────────────────────────────────────
            if step % 100 == 0:
                mode_str = "MANUAL" if manual_mode else "AUTO  "
                cells    = (grid.depth_count > 0).sum()
                print(f"[{mode_str}] step {step:5d} | "
                      f"pos ({pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:6.2f}) m | "
                      f"yaw {yaw:6.1f}° | cells: {cells:5d} | pts: {grid.n_points:7d}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    kb_ctrl.stop()

    if args.save:
        grid.save(args.save)

    if map3d is not None:
        map3d.update(grid)   # final full render before close
        map3d.close()

    if show:
        cv2.destroyAllWindows()

    # ── Summary statistics ────────────────────────────────────────────────────
    dm = grid.mean_depth
    valid = ~np.isnan(dm)
    if valid.any():
        print(f"\n=== Bathymetry summary ===")
        print(f"  Cells mapped : {valid.sum()} / {GRID_N * GRID_N}")
        print(f"  Depth range  : {dm[valid].min():.2f} m  to  {dm[valid].max():.2f} m")
        print(f"  Mean depth   : {dm[valid].mean():.2f} m")
        print(f"  Total points : {grid.n_points}")
    else:
        print("No bathymetry data collected.")


if __name__ == "__main__":
    main()
