"""MJPEG frame grabbers for the asset cameras.

Each camera is published on the sim host at its own port as
`multipart/x-mixed-replace; boundary=arcticframe`, carrying ordinary JPEGs of
roughly 30 KB. CAMERAS=ondemand in the sim's .env means a stream sleeps until
someone opens it and then costs about one server core while held, so grabbers
are opened lazily and closed when a sector goes quiet.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import requests

from .link import SIM_HOST

# Host-published camera ports, from GET :8090/api/assets.
CAM_PORTS = {
    "quadcopter": 8600,
    "fixed-wing": 8610,
    "tower-1": 8630,
    "tower-2": 8640,
}

_BOUNDARY = b"--arcticframe"
_SOI, _EOI = b"\xff\xd8", b"\xff\xd9"


@dataclass
class Frame:
    """One decoded frame, with the wall-clock time it was pulled off the wire."""

    image: np.ndarray       # BGR, HxWx3
    t: float
    seq: int

    @property
    def gray(self) -> np.ndarray:
        return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)


def stream_url(asset: str) -> str:
    return f"http://{SIM_HOST}:{CAM_PORTS[asset]}/stream"


def iter_jpegs(asset: str, timeout: float = 15.0, chunk: int = 32768):
    """Yield raw JPEG bytes from an asset's MJPEG stream.

    Scans for SOI/EOI markers rather than trusting Content-Length, which keeps
    it working if a frame is truncated mid-transfer.
    """
    with requests.get(stream_url(asset), stream=True, timeout=timeout) as r:
        r.raise_for_status()
        buf = bytearray()
        for block in r.iter_content(chunk_size=chunk):
            if not block:
                continue
            buf.extend(block)
            while True:
                i = buf.find(_SOI)
                if i < 0:
                    # No frame started yet; keep only a boundary's worth of tail.
                    if len(buf) > len(_BOUNDARY) * 4:
                        del buf[: -len(_BOUNDARY) * 4]
                    break
                j = buf.find(_EOI, i + 2)
                if j < 0:
                    del buf[:i]          # drop preamble, wait for the rest
                    break
                yield bytes(buf[i : j + 2])
                del buf[: j + 2]


class Grabber:
    """Background reader holding only the most recent frame.

    Detection always wants the freshest image, never a backlog, so this keeps a
    single slot rather than a queue -- a slow consumer drops frames instead of
    falling behind the world.
    """

    def __init__(self, asset: str):
        self.asset = asset
        self._frame: Frame | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self.dropped = 0
        self.errors = 0

    def start(self) -> "Grabber":
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{self.asset}")
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                for raw in iter_jpegs(self.asset):
                    if self._stop.is_set():
                        return
                    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        self.errors += 1
                        continue
                    self._seq += 1
                    with self._lock:
                        if self._frame is not None:
                            self.dropped += 1
                        self._frame = Frame(img, time.time(), self._seq)
            except Exception:
                self.errors += 1
                time.sleep(1.0)   # stream died; let the server settle, then retry

    def latest(self, max_age: float = 2.0) -> Frame | None:
        """Most recent frame, or None if the stream has gone stale."""
        with self._lock:
            f = self._frame
            self._frame = None
        if f is None or time.time() - f.t > max_age:
            return None
        return f

    def peek(self) -> Frame | None:
        with self._lock:
            return self._frame

    def wait(self, timeout: float = 5.0) -> Frame | None:
        """Block until a fresh frame lands."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            f = self.latest()
            if f is not None:
                return f
            time.sleep(0.01)
        return None


def probe(asset: str, n: int = 5) -> dict:
    """Pull a few frames and report what the camera actually delivers.

    The deck claims the towers are 640x360; tower-1 measures 1280x720. Nothing
    downstream should trust the documented figures, so every camera is measured.
    """
    sizes, t0, got = [], time.time(), 0
    for raw in iter_jpegs(asset):
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        sizes.append((img.shape[1], img.shape[0]))
        got += 1
        if got >= n:
            break
    dt = time.time() - t0
    return {
        "asset": asset,
        "frames": got,
        "size": sizes[0] if sizes else None,
        "consistent": len(set(sizes)) <= 1,
        "fps": (got / dt) if dt > 0 else 0.0,
    }
