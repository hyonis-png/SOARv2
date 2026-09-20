"""Pull satellite imagery for the site footprint, to locate the strait.

Tower placement needs to know where the navigable water actually is. The sim
builds its terrain from ArcticDEM for this exact footprint, so the coastline in
satellite imagery is the coastline in the sim.
"""
from __future__ import annotations
import math, os, sys
sys.path.insert(0, ".")
import requests
from soar import geo

def token() -> str:
    for line in open(".env"):
        if line.startswith("MAPBOX_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("MAPBOX_TOKEN missing from .env")

def bbox(extent_m: float = geo.SITE_EXTENT_M):
    half = extent_m / 2.0
    s, w = geo.to_latlon(-half, -half)
    n, e = geo.to_latlon(half, half)
    return w, s, e, n

if __name__ == "__main__":
    px = int(sys.argv[1]) if len(sys.argv) > 1 else 1280
    w, s, e, n = bbox()
    print(f"bbox lon {w:.6f}..{e:.6f}  lat {s:.6f}..{n:.6f}")
    url = (f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
           f"[{w},{s},{e},{n}]/{px}x{px}@2x")
    r = requests.get(url, params={"access_token": token(), "attribution": "false",
                                  "logo": "false"}, timeout=60)
    print("HTTP", r.status_code, len(r.content), "bytes", r.headers.get("content-type"))
    if r.status_code != 200:
        print(r.text[:400]); raise SystemExit(1)
    os.makedirs("data/site", exist_ok=True)
    out = "data/site/satellite.png"
    open(out, "wb").write(r.content)
    print("wrote", out)
