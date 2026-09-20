"""Geodesy and camera geometry for the Fort Ross site.

The Gazebo world is a flat local plane, so the flat-earth treatment used here is
exact rather than an approximation. Two consequences worth keeping in mind:

  * The horizon line in any camera image sits exactly on the camera's horizontal
    plane. Measuring a target's depression *relative to the detected horizon*
    therefore needs no pitch calibration at all -- it cancels.
  * Curvature drop at our 1500 m far clip would be ~0.18 m, which is below the
    size of the vessel we are chasing, so ignoring it costs nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Site, from GET :8090/api/site ---------------------------------------
SITE_NAME = "fort_ross"
SITE_LAT = 71.991960
SITE_LON = -94.822428
SITE_EXTENT_M = 6500.0
CONVERGENCE_DEG = -49.80479318596525  # EPSG:3413 grid convergence at the site

# Gazebo renders nothing past this. It is a hard sensor horizon, not a falloff.
FAR_CLIP_M = 1500.0

_MPD_LAT = 111132.0
_MPD_LON = 111320.0 * math.cos(math.radians(SITE_LAT))


# --- Local ENU ------------------------------------------------------------

def to_enu(lat: float, lon: float) -> tuple[float, float]:
    """Lat/lon -> (east, north) metres from site centre."""
    return ((lon - SITE_LON) * _MPD_LON, (lat - SITE_LAT) * _MPD_LAT)


def to_latlon(east: float, north: float) -> tuple[float, float]:
    """(east, north) metres from site centre -> lat/lon."""
    return (SITE_LAT + north / _MPD_LAT, SITE_LON + east / _MPD_LON)


def range_bearing(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Range in metres and true bearing in degrees from ENU point a to b."""
    de, dn = b[0] - a[0], b[1] - a[1]
    return math.hypot(de, dn), math.degrees(math.atan2(de, dn)) % 360.0


def project(origin: tuple[float, float], rng: float, bearing_deg: float) -> tuple[float, float]:
    """ENU point `rng` metres from `origin` along true bearing."""
    b = math.radians(bearing_deg)
    return (origin[0] + rng * math.sin(b), origin[1] + rng * math.cos(b))


# --- Camera model ---------------------------------------------------------

@dataclass(frozen=True)
class Camera:
    """A rectilinear pinhole camera, as Gazebo renders it."""

    name: str
    hfov_deg: float
    vfov_deg: float
    width: int
    height: int

    @property
    def fx(self) -> float:
        """Focal length in pixels, horizontal."""
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def fy(self) -> float:
        """Focal length in pixels, vertical."""
        return (self.height / 2.0) / math.tan(math.radians(self.vfov_deg) / 2.0)

    def yaw_offset_deg(self, x: float) -> float:
        """Horizontal angle of image column `x` off the optical axis."""
        return math.degrees(math.atan2(x - self.width / 2.0, self.fx))

    def pitch_offset_deg(self, y: float) -> float:
        """Angle of image row `y` below the optical axis (positive = below)."""
        return math.degrees(math.atan2(y - self.height / 2.0, self.fy))


# Optics. The field-of-view figures come from the deck and check out -- fx and
# fy agree to within a percent at each camera, so the pixels are square. The
# RESOLUTIONS in the deck are stale: it claims towers are 640x360 and the quad
# 640x480, but the streams actually deliver what is written here, measured by
# soar.video.probe against the live sim.
#
# All four cameras publish at ~10 fps, which matters more than it looks. A 3 m/s
# vessel at 1000 m subtends 0.17 deg/s, or about 0.33 px per frame at the tower
# focal length -- sub-pixel per-frame motion. Any motion detector here has to
# integrate over a dwell, not over a frame pair.
STREAM_FPS = 10.0

CAMERAS = {
    "quadcopter": Camera("quadcopter", 114.6, 99.4, 960, 720),
    "fixed-wing": Camera("fixed-wing", 69.0, 42.6, 1280, 720),
    "tower-1": Camera("tower-1", 60.0, 36.1, 1280, 720),
    "tower-2": Camera("tower-2", 60.0, 36.1, 1280, 720),
}


def pixels_per_frame(cam: "Camera", speed_mps: float, rng: float, fps: float = STREAM_FPS) -> float:
    """Cross-track image motion in px/frame for a target at `rng` metres.

    Sanity check on whether a dwell is long enough to see anything move.
    """
    if rng <= 0:
        return float("inf")
    return (speed_mps / rng) * cam.fy / fps


# --- Horizon-depression ranging ------------------------------------------
#
# The towers cannot triangulate: they sit 3846 m apart with a 1500 m far clip,
# so their coverage disks are disjoint by 846 m and they never see the vessel
# at the same moment. Range from a single tower therefore has to come from the
# geometry of the water plane.

def true_horizon_row(cam: Camera, pitch_deg: float) -> float:
    """Image row of the TRUE horizon, from the camera's pitch.

    This must come from attitude, not from the picture. Bellot Strait is
    enclosed: the sky/water edge you can see is the FAR SHORE, which stands at a
    finite distance and therefore sits BELOW the true horizontal. Ranging off
    that edge understates the depression angle badly -- measured against a real
    contact it put a 1.3 km vessel at 18 km.
    """
    return cam.height / 2.0 + cam.fy * math.tan(math.radians(pitch_deg))


def range_from_horizon(cam: Camera, height_m: float, y_target: float,
                       pitch_deg: float) -> float | None:
    """Range to a waterline contact, from its depression below the true horizon.

    Flat world, so range = height / tan(depression) exactly.

    Verified against a real detection: tower-1 at 116.8 m AMSL, vessel 96 px
    below the pitch-derived horizon, gives 1330 m -- inside the 1500 m far clip,
    and the range rate across the dwell came out at ~2.8 m/s against the sim's
    configured SHIP_SPEED of 3.0.

    Returns None when the contact is at or above the horizon, which means it is
    not floating and should be discarded.
    """
    y_h = true_horizon_row(cam, pitch_deg)
    dy = y_target - y_h
    if dy <= 0:
        return None
    theta = math.atan2(dy, cam.fy)
    if theta <= 0:
        return None
    return height_m / math.tan(theta)


def max_range_row(cam: Camera, height_m: float, pitch_deg: float,
                  far: float = FAR_CLIP_M) -> float:
    """Image row at the far clip. Anything ABOVE this is out of render range."""
    y_h = true_horizon_row(cam, pitch_deg)
    return y_h + cam.fy * math.tan(math.atan2(height_m, far))


def range_error_per_pixel(cam: Camera, height_m: float, rng: float) -> float:
    """Range uncertainty contributed by one pixel of vertical error.

    Grows as range squared, which is the quantitative reason the quadcopter's
    nadir fix is the accurate one and the tower fix is the cue.
    """
    if rng <= 0:
        return float("inf")
    theta = math.atan2(height_m, rng)
    return (height_m / math.sin(theta) ** 2) * (1.0 / cam.fy)


def bearing_from_pixel(cam: Camera, x: float, yaw_deg: float) -> float:
    """True bearing of image column `x` given the camera's own yaw."""
    return (yaw_deg + cam.yaw_offset_deg(x)) % 360.0


# --- Nadir projection -----------------------------------------------------

def nadir_fix(
    cam: Camera,
    origin: tuple[float, float],
    alt_m: float,
    x: float,
    y: float,
    yaw_deg: float,
) -> tuple[float, float]:
    """Ground position of pixel (x, y) from a downward-looking camera.

    Error here is roughly altitude x angular error, with no slant-range
    multiplier -- which is why the quadcopter holds station directly overhead.
    """
    # Metres on the ground per pixel at nadir.
    east_off = (x - cam.width / 2.0) * alt_m / cam.fx
    north_off = -(y - cam.height / 2.0) * alt_m / cam.fy
    yaw = math.radians(yaw_deg)
    e = east_off * math.cos(yaw) + north_off * math.sin(yaw)
    n = -east_off * math.sin(yaw) + north_off * math.cos(yaw)
    return (origin[0] + e, origin[1] + n)
