"""Choose tower sites: maximise visible strait water in an assigned sector.

Keeps the thirds architecture -- tower-1 owns the first third of the strait,
tower-2 the last third, the fixed-wing patrols the middle -- but picks WHERE
each tower stands by measurement rather than by eye.

Scoring is deliberately strict. A water cell counts for a candidate site only if
it is inside the 1500 m far clip (beyond that Gazebo renders nothing at all) AND
has genuine line of sight over the intervening terrain. Height helps twice: it
buys line of sight over the shore banks, and it flattens the horizon-depression
geometry that turns a bearing into a range.
"""
from __future__ import annotations
import math, sys
sys.path.insert(0, ".")
import numpy as np
from scipy import ndimage
from soar import geo

FAR = geo.FAR_CLIP_M
MAST_M = 3.0          # camera height above local ground
CLEAR_M = 2.0         # required clearance over intervening terrain
GRID = 60.0           # metres between candidate sites / water cells


def load_dem():
    dem = np.load("data/site/dem.npy")
    lat_tl, lon_tl, lat_br, lon_br = np.load("data/site/dem_bounds.npy")
    h, w = dem.shape
    lats = np.linspace(lat_tl, lat_br, h)
    lons = np.linspace(lon_tl, lon_br, w)
    return dem, lats, lons


def build_grid():
    """Resample the DEM onto a regular ENU grid covering the site."""
    dem, lats, lons = load_dem()
    half = geo.SITE_EXTENT_M / 2.0
    n = int(geo.SITE_EXTENT_M / GRID) + 1
    es = np.linspace(-half, half, n)
    ns = np.linspace(-half, half, n)
    EE, NN = np.meshgrid(es, ns)
    LAT = geo.SITE_LAT + NN / 111132.0
    LON = geo.SITE_LON + EE / (111320.0 * math.cos(math.radians(geo.SITE_LAT)))
    ri = np.clip(np.searchsorted(-lats, -LAT), 0, dem.shape[0] - 1)
    ci = np.clip(np.searchsorted(lons, LON), 0, dem.shape[1] - 1)
    return EE, NN, dem[ri, ci]


def strait_mask(Z):
    """Largest connected body of water -- the strait, not the inland ponds."""
    water = Z <= 0.0
    lab, k = ndimage.label(water)
    if k == 0:
        raise SystemExit("no water found")
    sizes = ndimage.sum(water, lab, range(1, k + 1))
    return lab == (1 + int(np.argmax(sizes)))


def sectors(EE, NN, sea):
    """Split the strait into thirds along its own long axis."""
    pts = np.stack([EE[sea], NN[sea]], 1)
    c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
    axis = vt[0]
    if axis[0] < 0:
        axis = -axis                      # point the axis east, for readability
    t = (pts - c) @ axis
    lo, hi = t.min(), t.max()
    edges = [lo, lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3, hi]
    sec = np.digitize(t, edges[1:3])      # 0,1,2
    return pts, c, axis, t, sec, edges


def visible(cx, cy, cz, tx, ty, Zgrid, EE, NN, k=24):
    """Vectorised LOS from one site to many water cells."""
    f = np.linspace(0.08, 0.96, k)[None, :]
    px = cx + (tx[:, None] - cx) * f
    py = cy + (ty[:, None] - cy) * f
    e0, n0 = EE[0, 0], NN[0, 0]
    ci = np.clip(((px - e0) / GRID).round().astype(int), 0, EE.shape[1] - 1)
    ri = np.clip(((py - n0) / GRID).round().astype(int), 0, EE.shape[0] - 1)
    terr = Zgrid[ri, ci]
    ray = cz + (0.0 - cz) * f
    return (ray > terr + CLEAR_M).all(axis=1)


def best_site(EE, NN, Z, sea, target, exclude=None, min_elev=4.0, max_shore=900.0):
    """Land site maximising visible water cells within the target sector."""
    tx, ty = EE[target], NN[target]
    # Candidate sites: dry land, near enough to the water to be useful.
    land = (Z > min_elev) & ~sea
    dist_to_water = ndimage.distance_transform_edt(~sea) * GRID
    cand = land & (dist_to_water < max_shore)
    ce, cn, cz = EE[cand], NN[cand], Z[cand] + MAST_M

    best = None
    for i in range(len(ce)):
        d = np.hypot(tx - ce[i], ty - cn[i])
        near = d < FAR
        if near.sum() == 0:
            continue
        if best is not None and near.sum() <= best[0]:
            continue                       # cannot beat the incumbent
        vis = visible(ce[i], cn[i], cz[i], tx[near], ty[near], Z, EE, NN)
        score = int(vis.sum())
        if best is None or score > best[0]:
            best = (score, ce[i], cn[i], cz[i], Z[cand][i], int(near.sum()))
    return best


if __name__ == "__main__":
    EE, NN, Z = build_grid()
    sea = strait_mask(Z)
    pts, c, axis, t, sec, edges = sectors(EE, NN, sea)
    bearing = math.degrees(math.atan2(axis[0], axis[1])) % 360.0
    total = int(sea.sum())
    print(f"grid {EE.shape} @ {GRID:.0f} m | strait cells {total} "
          f"({total*GRID*GRID/1e6:.1f} km2)")
    print(f"strait axis bearing {bearing:.1f} deg, length {edges[-1]-edges[0]:.0f} m\n")

    secmask = np.zeros(EE.shape, int) - 1
    secmask[sea] = sec
    names = ["first third (W)", "middle third", "last third (E)"]
    for i in range(3):
        m = secmask == i
        print(f"  sector {i} {names[i]:16s}: {int(m.sum()):5d} cells "
              f"({m.sum()*GRID*GRID/1e6:5.2f} km2)")

    print("\nsiting towers (LOS-checked, 1500 m far clip)...")
    out = {}
    for i, role in ((0, "tower-1"), (2, "tower-2")):
        tgt = secmask == i
        b = best_site(EE, NN, Z, sea, tgt)
        score, e, n, camz, ground, inrange = b
        lat, lon = geo.to_latlon(e, n)
        cov = 100.0 * score / max(1, int(tgt.sum()))
        out[role] = (lat, lon, ground, score, cov)
        print(f"\n  {role} -> sector {i} ({names[i]})")
        print(f"    ENU        {e:8.0f} E {n:8.0f} N")
        print(f"    lat,lon    {lat:.6f},{lon:.6f}")
        print(f"    ground     {ground:.1f} m  (camera {camz:.1f} m AMSL)")
        print(f"    in range   {inrange} cells | VISIBLE {score} cells "
              f"= {cov:.0f}% of sector")
    np.save("data/site/secmask.npy", secmask)
    np.save("data/site/grid.npy", np.stack([EE, NN, Z]))
    print("\nASSET lines for .env:")
    for role in ("tower-1", "tower-2"):
        lat, lon, *_ = out[role]
        print(f"  tower,{role},{lat:.6f},{lon:.6f}")
