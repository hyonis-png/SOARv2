"""Submit contacts to the scoring API."""
from __future__ import annotations
import requests
from .link import SIM_HOST
BASE = f"http://{SIM_HOST}:8010"

def post_track(name, lat, lon, heading=None, speed=None, timeout=15):
    p = {"name": name, "lat": round(lat, 6), "lon": round(lon, 6)}
    if heading is not None: p["heading"] = round(heading, 1)
    if speed is not None:   p["speed"] = round(speed, 2)
    r = requests.post(f"{BASE}/api/tracks", json=p, timeout=timeout)
    r.raise_for_status(); return r.json()

def list_tracks(timeout=15):
    return requests.get(f"{BASE}/api/tracks", timeout=timeout).json()
