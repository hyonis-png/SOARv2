"""Jointly optimise both tower sites for maximum union coverage of the strait.

Siting the towers one at a time is greedy and leaves coverage on the table: what
matters is the UNION of what the pair can see. Because the 1500 m far clip is the
binding constraint (not line of sight), the best sites hug the water -- and an
island in mid-channel beats a hilltop, because a disk centred on water is all
water while a disk centred on shore is half land.
"""
from __future__ import annotations
import math, sys
sys.path.insert(0, ".")
import numpy as np
from scipy import ndimage
from eval.site_towers import build_grid, strait_mask, sectors, visible, MAST_M, GRID
from soar import geo

EE, NN, Z = build_grid()
sea = strait_mask(Z)
pts, c, axis, t, sec, edges = sectors(EE, NN, sea)
tx, ty = EE[sea], NN[sea]
nwater = len(tx)
cell_km2 = GRID * GRID / 1e6

# Along-strait coordinate of every water cell, normalised 0..1 west->east.
s_water = ((np.stack([tx, ty], 1) - c) @ axis - edges[0]) / (edges[-1] - edges[0])

# --- candidate sites: any dry land near water, islands included ---
land = (Z > 3.0) & ~sea
dist = ndimage.distance_transform_edt(~sea) * GRID
cand = land & (dist < 700.0)
ce, cn, cg = EE[cand], NN[cand], Z[cand]
print(f"{len(ce)} candidate sites | {nwater} water cells ({nwater*cell_km2:.1f} km2)")

# --- visibility set for each candidate ---
vis_sets = np.zeros((len(ce), nwater), bool)
for i in range(len(ce)):
    d = np.hypot(tx - ce[i], ty - cn[i])
    near = d < geo.FAR_CLIP_M
    if not near.any():
        continue
    v = visible(ce[i], cn[i], cg[i] + MAST_M, tx[near], ty[near], Z, EE, NN)
    idx = np.flatnonzero(near)
    vis_sets[i, idx[v]] = True

counts = vis_sets.sum(1)
print(f"best single site: {counts.max()} cells ({counts.max()*cell_km2:.1f} km2, "
      f"{100*counts.max()/nwater:.0f}% of strait)")

# Keep the strongest candidates; a weak site cannot be in the optimal pair when
# even the union bound cannot beat the incumbent.
keep = np.argsort(-counts)[:400]
packed = np.packbits(vis_sets[keep], axis=1)

# --- exhaustive best pair ---
best = (-1, None, None)
for a in range(len(keep)):
    union = np.bitwise_or(packed[a], packed[a + 1:])
    if len(union) == 0:
        continue
    tot = np.unpackbits(union, axis=1).sum(1)
    j = int(np.argmax(tot))
    if tot[j] > best[0]:
        best = (int(tot[j]), a, a + 1 + j)

score, ia, ib = best
A, B = keep[ia], keep[ib]
# Order west -> east so tower-1 owns the early strait.
if ((np.array([ce[A], cn[A]]) - c) @ axis) > ((np.array([ce[B], cn[B]]) - c) @ axis):
    A, B = B, A

print(f"\nbest PAIR: {score} cells ({score*cell_km2:.1f} km2) "
      f"= {100*score/nwater:.0f}% of strait\n")

out = {}
for role, k in (("tower-1", A), ("tower-2", B)):
    lat, lon = geo.to_latlon(ce[k], cn[k])
    frac = ((np.array([ce[k], cn[k]]) - c) @ axis - edges[0]) / (edges[-1] - edges[0])
    on_island = dist[cand][k] if False else None
    seen = vis_sets[k]
    print(f"{role}: {lat:.6f},{lon:.6f}")
    print(f"   ENU {ce[k]:8.0f} E {cn[k]:8.0f} N | ground {cg[k]:5.1f} m "
          f"| {int(seen.sum())} cells ({seen.sum()*cell_km2:.1f} km2)")
    print(f"   sits at {frac*100:.0f}% along the strait (west->east)")
    # what share of each third does it see?
    for si, nm in ((0, "W third"), (1, "mid third"), (2, "E third")):
        m = (s_water >= si/3) & (s_water < (si+1)/3 + (1e-9 if si == 2 else 0))
        if m.sum():
            print(f"      {nm:10s}: {100*(seen & m).sum()/m.sum():3.0f}% covered")
    out[role] = (lat, lon)

union = vis_sets[A] | vis_sets[B]
print(f"\nunion covers {100*union.sum()/nwater:.0f}% of strait; "
      f"{int((~union).sum())} cells ({(~union).sum()*cell_km2:.1f} km2) left for the fixed-wing")
for si, nm in ((0, "W third"), (1, "mid third"), (2, "E third")):
    m = (s_water >= si/3) & (s_water < (si+1)/3 + (1e-9 if si == 2 else 0))
    print(f"   {nm:10s}: {100*(union & m).sum()/m.sum():3.0f}% by towers")
np.save("data/site/tower_vis.npy", np.stack([vis_sets[A], vis_sets[B]]))
print("\nASSET lines for .env:")
for role in ("tower-1", "tower-2"):
    print(f"  ASSET_?={role.replace('tower-','tower,tower-')},{out[role][0]:.6f},{out[role][1]:.6f}")
