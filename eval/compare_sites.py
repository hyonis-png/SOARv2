"""Score any tower position against the sector it is meant to own."""
from __future__ import annotations
import math, sys
sys.path.insert(0, ".")
import numpy as np
sys.argv = sys.argv
from eval.site_towers import build_grid, strait_mask, sectors, visible, MAST_M, GRID
from soar import geo

EE, NN, Z = build_grid()
sea = strait_mask(Z)
pts, c, axis, t, sec, edges = sectors(EE, NN, sea)
secmask = np.zeros(EE.shape, int) - 1
secmask[sea] = sec

def score_at(lat, lon, sector):
    e, n = geo.to_enu(lat, lon)
    ci = int(np.clip(round((e - EE[0, 0]) / GRID), 0, EE.shape[1] - 1))
    ri = int(np.clip(round((n - NN[0, 0]) / GRID), 0, EE.shape[0] - 1))
    ground = float(Z[ri, ci])
    tgt = secmask == sector
    tx, ty = EE[tgt], NN[tgt]
    d = np.hypot(tx - e, ty - n)
    near = d < geo.FAR_CLIP_M
    if near.sum() == 0:
        return ground, 0, 0, int(tgt.sum())
    vis = visible(e, n, ground + MAST_M, tx[near], ty[near], Z, EE, NN)
    return ground, int(vis.sum()), int(near.sum()), int(tgt.sum())

def whole_strait(lat, lon):
    e, n = geo.to_enu(lat, lon)
    ci = int(np.clip(round((e - EE[0, 0]) / GRID), 0, EE.shape[1] - 1))
    ri = int(np.clip(round((n - NN[0, 0]) / GRID), 0, EE.shape[0] - 1))
    tx, ty = EE[sea], NN[sea]
    d = np.hypot(tx - e, ty - n)
    near = d < geo.FAR_CLIP_M
    if near.sum() == 0:
        return 0
    return int(visible(e, n, Z[ri, ci] + MAST_M, tx[near], ty[near], Z, EE, NN).sum())

CUR = {"tower-1": (71.980671, -94.853711), "tower-2": (72.011778, -94.804721)}
NEW = {"tower-1": (71.994668, -94.897628), "tower-2": (71.990877, -94.745480)}

print(f"{'':10s} {'site':22s} {'ground':>7s} {'in-range':>9s} {'VISIBLE':>8s} {'sector%':>8s}")
print("-" * 70)
tot_cur = tot_new = 0
for role, s in (("tower-1", 0), ("tower-2", 2)):
    for label, tbl in (("current", CUR), ("optimised", NEW)):
        lat, lon = tbl[role]
        g, vis, near, tgt = score_at(lat, lon, s)
        if label == "current": tot_cur += vis
        else: tot_new += vis
        print(f"{role if label=='current' else '':10s} {label+' '+f'{lat:.5f},{lon:.5f}':22s} "
              f"{g:6.0f}m {near:9d} {vis:8d} {100*vis/max(1,tgt):7.0f}%")
    print()

strait_total = int(sea.sum())
cur_union = sum(whole_strait(*CUR[r]) for r in CUR)
new_union = sum(whole_strait(*NEW[r]) for r in NEW)
cell_km2 = GRID * GRID / 1e6
print(f"strait total                 {strait_total:5d} cells ({strait_total*cell_km2:.1f} km2)")
print(f"towers as placed now         {cur_union:5d} cells ({cur_union*cell_km2:.1f} km2) "
      f"= {100*cur_union/strait_total:.0f}% of strait")
print(f"towers optimised             {new_union:5d} cells ({new_union*cell_km2:.1f} km2) "
      f"= {100*new_union/strait_total:.0f}% of strait")
print(f"\ngain: {new_union/max(1,cur_union):.1f}x more visible strait water")
