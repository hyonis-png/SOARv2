"""Render the siting decision as a map: strait, sectors, towers, coverage."""
from __future__ import annotations
import math, sys
sys.path.insert(0, ".")
import numpy as np, cv2
from eval.site_towers import build_grid, strait_mask, sectors, visible, MAST_M, GRID
from soar import geo

EE, NN, Z = build_grid()
sea = strait_mask(Z)
pts, c, axis, t, sec, edges = sectors(EE, NN, sea)

S = 1100
def to_px(e, n):
    f = S / geo.SITE_EXTENT_M
    return int((e + geo.SITE_EXTENT_M/2) * f), int(S - (n + geo.SITE_EXTENT_M/2) * f)

img = np.zeros((S, S, 3), np.uint8)
# terrain shading
zc = np.clip(Z, 0, 300) / 300.0
# grid rows run south->north; image rows run north->south, so flip.
terr = cv2.resize(np.flipud((zc*150+40).astype(np.uint8)), (S, S), interpolation=cv2.INTER_LINEAR)
img[:, :, 0] = terr//2; img[:, :, 1] = (terr*0.75).astype(np.uint8); img[:, :, 2] = terr
# water
wm = cv2.resize(np.flipud(sea).astype(np.uint8)*255, (S, S), interpolation=cv2.INTER_NEAREST)
img[wm > 0] = (70, 35, 12)

# sector bands along the strait axis
secmask = np.zeros(EE.shape, int) - 1; secmask[sea] = sec
cols = [(120, 70, 30), (60, 90, 40), (40, 60, 110)]
for i in range(3):
    m = cv2.resize(np.flipud(secmask == i).astype(np.uint8)*255, (S, S), interpolation=cv2.INTER_NEAREST)
    img[m > 0] = cols[i]

CUR = {"tower-1": (71.980671, -94.853711), "tower-2": (72.011778, -94.804721)}
NEW = {"tower-1": (71.981129, -94.853907), "tower-2": (71.988169, -94.778707)}

# coverage disks for the new siting (far clip)
ov = img.copy()
for lat, lon in NEW.values():
    e, n = geo.to_enu(lat, lon)
    cv2.circle(ov, to_px(e, n), int(geo.FAR_CLIP_M * S / geo.SITE_EXTENT_M), (200, 220, 255), -1)
img = cv2.addWeighted(ov, 0.16, img, 0.84, 0)

for lat, lon in CUR.values():
    p = to_px(*geo.to_enu(lat, lon))
    cv2.drawMarker(img, p, (60, 60, 200), cv2.MARKER_TILTED_CROSS, 18, 2)
for role, (lat, lon) in NEW.items():
    e, n = geo.to_enu(lat, lon)
    p = to_px(e, n)
    cv2.circle(img, p, int(geo.FAR_CLIP_M * S / geo.SITE_EXTENT_M), (255, 235, 180), 1)
    cv2.circle(img, p, 7, (120, 255, 120), -1); cv2.circle(img, p, 7, (0, 0, 0), 2)
    cv2.putText(img, role, (p[0]+11, p[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 4)
    cv2.putText(img, role, (p[0]+11, p[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140,255,140), 1)

# strait axis + third marks
for f, lbl in ((0.0,""), (1/3,"1/3"), (2/3,"2/3"), (1.0,"")):
    q = c + axis * (edges[0] + f*(edges[-1]-edges[0]))
    p = to_px(*q)
    if lbl:
        cv2.line(img, (p[0], p[1]-26), (p[0], p[1]+26), (255,255,255), 2)
        cv2.putText(img, lbl, (p[0]-14, p[1]-32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

leg = ["red X = current   green = optimised   ring = 1500m far clip",
       "bands = W / mid / E thirds of the strait"]
for i, s_ in enumerate(leg):
    cv2.putText(img, s_, (10, S-34+18*i), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0,0,0), 3)
    cv2.putText(img, s_, (10, S-34+18*i), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 1)
cv2.imwrite("data/site/plan.jpg", img)
print("wrote data/site/plan.jpg")
