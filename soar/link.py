"""MAVLink transport to the four ArduPilot SITL assets.

Two things about this sim that the documentation gets wrong, both verified by
probing the live host:

  * The 10.23.0.x container addresses in the .env are NOT routable over the
    WireGuard tunnel -- only the host's published ports are. The sim's own
    control panel says as much in a comment: the `arctic` bridge is internal to
    the Docker host.
  * The TCP ports (5760/5770/5790/5800) accept a connection and then stay
    silent, because something in-container already holds them. UDP is the live
    path: send anything and ArduPilot starts streaming back. Hence `udpout`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from pymavlink import mavutil

SIM_HOST = "10.99.11.1"

# Host-published UDP ports, from GET :8090/api/assets.
ENDPOINTS = {
    "quadcopter": 14550,
    "fixed-wing": 14560,
    "tower-1": 14580,
    "tower-2": 14590,
}

# ArduPilot custom mode numbers we actually use.
COPTER_MODES = {"STABILIZE": 0, "GUIDED": 4, "LOITER": 5, "RTL": 6, "LAND": 9}
PLANE_MODES = {"MANUAL": 0, "CIRCLE": 1, "GUIDED": 15, "TAKEOFF": 13, "LOITER": 12}
TRACKER_MODES = {"MANUAL": 0, "SCAN": 2, "SERVO_TEST": 3, "AUTO": 10, "INITIALISING": 16}


@dataclass
class Telemetry:
    """Latest known state of one asset. Written by the reader thread."""

    lat: float | None = None
    lon: float | None = None
    alt_m: float | None = None          # AMSL
    rel_alt_m: float | None = None      # above home
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    groundspeed: float | None = None
    armed: bool = False
    mode: int | None = None
    servo: dict[int, int] = field(default_factory=dict)
    last_update: float = 0.0

    @property
    def fresh(self) -> bool:
        return time.time() - self.last_update < 3.0


class Asset:
    """One vehicle. Owns a reader thread that keeps `telem` current."""

    def __init__(self, name: str, host: str = SIM_HOST, port: int | None = None):
        self.name = name
        self.port = port if port is not None else ENDPOINTS[name]
        self.telem = Telemetry()
        self._conn: mavutil.mavfile | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # --- lifecycle ---

    def connect(self, timeout: float = 10.0) -> bool:
        """Open the link and wait for the first heartbeat."""
        self._conn = mavutil.mavlink_connection(
            f"udpout:{SIM_HOST}:{self.port}", source_system=255, source_component=190
        )
        # udpout stays silent until we speak first.
        self._conn.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
        )
        hb = self._conn.wait_heartbeat(timeout=timeout)
        if hb is None:
            return False
        self._request_streams()
        self._thread = threading.Thread(target=self._reader, daemon=True, name=f"mav-{self.name}")
        self._thread.start()
        return True

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._conn:
            self._conn.close()

    def _request_streams(self, rate_hz: int = 10) -> None:
        assert self._conn is not None
        self._conn.mav.request_data_stream_send(
            self._conn.target_system,
            self._conn.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL,
            rate_hz,
            1,
        )

    # --- reader thread ---

    def _reader(self) -> None:
        assert self._conn is not None
        last_hb = 0.0
        while not self._stop.is_set():
            # ArduPilot drops a udpout GCS that goes quiet, so keep talking.
            now = time.time()
            if now - last_hb > 1.0:
                try:
                    self._conn.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                    )
                except Exception:
                    pass
                last_hb = now

            msg = self._conn.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue
            self._ingest(msg)

    def _ingest(self, msg) -> None:
        t = msg.get_type()
        with self._lock:
            tel = self.telem
            if t == "GLOBAL_POSITION_INT":
                tel.lat = msg.lat / 1e7
                tel.lon = msg.lon / 1e7
                tel.alt_m = msg.alt / 1000.0
                tel.rel_alt_m = msg.relative_alt / 1000.0
            elif t == "ATTITUDE":
                import math
                tel.roll_deg = math.degrees(msg.roll)
                tel.pitch_deg = math.degrees(msg.pitch)
                tel.yaw_deg = math.degrees(msg.yaw) % 360.0
            elif t == "VFR_HUD":
                tel.groundspeed = msg.groundspeed
            elif t == "HEARTBEAT":
                tel.armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                tel.mode = msg.custom_mode
            elif t == "SERVO_OUTPUT_RAW":
                for i in range(1, 9):
                    v = getattr(msg, f"servo{i}_raw", None)
                    if v is not None:
                        tel.servo[i] = v
            tel.last_update = time.time()

    def snapshot(self) -> Telemetry:
        with self._lock:
            import copy
            return copy.copy(self.telem)

    # --- commands ---

    def set_mode(self, mode_num: int) -> None:
        assert self._conn is not None
        self._conn.mav.set_mode_send(
            self._conn.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_num,
        )

    def arm(self, arm: bool = True) -> None:
        assert self._conn is not None
        self._conn.mav.command_long_send(
            self._conn.target_system, self._conn.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1 if arm else 0, 0, 0, 0, 0, 0, 0,
        )

    def takeoff(self, alt_m: float) -> None:
        """Guided takeoff.

        The deck warns guided mode "only runs for 3 secs", so callers must issue
        arm and takeoff back to back with no waiting in between.
        """
        assert self._conn is not None
        self._conn.mav.command_long_send(
            self._conn.target_system, self._conn.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, alt_m,
        )

    def goto(self, lat: float, lon: float, alt_m: float) -> None:
        """Guided reposition, equivalent to MAVProxy's `guided <lat> <lon> <alt>`."""
        assert self._conn is not None
        self._conn.mav.set_position_target_global_int_send(
            0, self._conn.target_system, self._conn.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,
            int(lat * 1e7), int(lon * 1e7), alt_m,
            0, 0, 0, 0, 0, 0, 0, 0,
        )

    def set_servo(self, channel: int, pwm: int) -> None:
        """Drive a servo directly. Towers pan on SERVO1 and tilt on SERVO2."""
        assert self._conn is not None
        self._conn.mav.command_long_send(
            self._conn.target_system, self._conn.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO, 0,
            channel, pwm, 0, 0, 0, 0, 0,
        )


def connect_all(names: list[str] | None = None, timeout: float = 10.0) -> dict[str, Asset]:
    """Connect to several assets in parallel; returns only those that answered."""
    names = names or list(ENDPOINTS)
    assets = {n: Asset(n) for n in names}
    results: dict[str, bool] = {}

    def _go(n: str) -> None:
        try:
            results[n] = assets[n].connect(timeout=timeout)
        except Exception:
            results[n] = False

    threads = [threading.Thread(target=_go, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 2.0)
    return {n: a for n, a in assets.items() if results.get(n)}
