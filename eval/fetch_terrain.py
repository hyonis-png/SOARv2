"""Fetch Mapbox Terrain-RGB for the site and decode it to a metre DEM.

Elevation is a better basis for tower siting than image colour: it defines
land/water without ambiguity (sea level is 0), it gives each candidate site's
height above the water -- which is exactly what horizon-depression ranging
needs -- and it supports line-of-sight checks against intervening ridges.
"""
from __future__ import annotations
import io, math, os, sys
sys.path.insert(0, ".")
import numpy as np
import requests
from PIL import Image
from soar import geo

Z = 14  # ~2.9 m/px at 72N -- finer than the sim's own 6.3 m/sample terrain


def token() -> str:
    for line in open(".env"):
        if line.startswith("MAPBOX_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("MAPBOX_TOKEN missing from .env")


def deg2tile(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def tile2deg(x, y, z):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


if __name__ == "__main__":
    tok = token()
    half = geo.SITE_EXTENT_M / 2.0
    s, w = geo.to_latlon(-half, -half)
    n_, e = geo.to_latlon(half, half)

    x0f, y0f = deg2tile(n_, w, Z)          # top-left
    x1f, y1f = deg2tile(s, e, Z)           # bottom-right
    x0, y0, x1, y1 = int(x0f), int(y0f), int(x1f), int(y1f)
    print(f"z={Z} tiles x {x0}..{x1} y {y0}..{y1} "
          f"({(x1-x0+1)*(y1-y0+1)} tiles)")

    W, H = (x1 - x0 + 1) * 256, (y1 - y0 + 1) * 256
    canvas = np.zeros((H, W, 3), np.uint8)
    for tx in range(x0, x1 + 1):
        for ty in range(y0, y1 + 1):
            url = f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{Z}/{tx}/{ty}.pngraw"
            r = requests.get(url, params={"access_token": tok}, timeout=60)
            if r.status_code != 200:
                print(f"  tile {tx},{ty}: HTTP {r.status_code} {r.text[:120]}")
                continue
            im = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
            canvas[(ty - y0) * 256:(ty - y0 + 1) * 256,
                   (tx - x0) * 256:(tx - x0 + 1) * 256] = im

    # Mapbox Terrain-RGB decoding.
    R, G, B = canvas[:, :, 0].astype(np.float64), canvas[:, :, 1].astype(np.float64), canvas[:, :, 2].astype(np.float64)
    dem = -10000.0 + (R * 65536.0 + G * 256.0 + B) * 0.1

    os.makedirs("data/site", exist_ok=True)
    np.save("data/site/dem.npy", dem.astype(np.float32))
    lat_tl, lon_tl = tile2deg(x0, y0, Z)
    lat_br, lon_br = tile2deg(x1 + 1, y1 + 1, Z)
    np.save("data/site/dem_bounds.npy",
            np.array([lat_tl, lon_tl, lat_br, lon_br], np.float64))
    print(f"DEM {dem.shape}  elev min={dem.min():.1f} max={dem.max():.1f} m")
    print(f"bounds  TL {lat_tl:.6f},{lon_tl:.6f}   BR {lat_br:.6f},{lon_br:.6f}")
    print(f"water (elev<=0): {(dem <= 0).mean()*100:.1f}% of tile area")
