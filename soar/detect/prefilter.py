"""Stage 0: find the water, find the horizon, propose candidate ROIs.

This stage is deliberately high-recall and cheap. Its job is to hand the
expensive fly-vision stage a short list of places worth looking, and -- more
importantly -- to establish the horizon row, which is what turns a detection
into a range (see soar.geo.range_from_horizon).

The sea in this sim is covered in bright specular flecks. Stage 0 does NOT try
to tell them from a vessel; that is exactly the discrimination Stage 1 exists
for. Here we only insist a candidate is dark-on-water and below the horizon.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Candidate:
    """A place worth foveating on, in pixel coordinates."""

    x: float
    y: float
    area: float
    contrast: float     # how much darker than local sea, 0..1


@dataclass(frozen=True)
class Scene:
    horizon_y: float | None
    water_mask: np.ndarray          # uint8, 255 where sea
    candidates: list[Candidate]


def find_horizon(gray: np.ndarray) -> float | None:
    """Row of the sky/sea boundary.

    In a flat world the horizon lies exactly on the camera's horizontal plane,
    so this row is a free pitch reference -- no servo calibration needed. Found
    by the strongest sustained vertical gradient, which is robust here because
    sky is bright and uniform while the sea is dark.
    """
    h, w = gray.shape
    col = cv2.GaussianBlur(gray, (0, 0), 3).mean(axis=1)
    d = np.diff(col.astype(np.float32))
    # Sky above water means a strong negative step going down the image.
    k = int(np.argmin(d))
    if -d[k] < 6.0:                 # no clear sky/sea step (looking at land, or fogged out)
        return None
    return float(k + 1)


def water_mask(bgr: np.ndarray, horizon_y: float | None) -> np.ndarray:
    """Pixels that are plausibly open sea.

    Sea in this render is dark and blue-dominant; terrain is warm and bright.
    Anything above the horizon is excluded outright, which is the single
    cheapest false-positive filter available -- the vessel is on the water, so
    nothing above the horizon can be it.
    """
    h, w = bgr.shape[:2]
    b, g, r = bgr[:, :, 0].astype(np.int16), bgr[:, :, 1].astype(np.int16), bgr[:, :, 2].astype(np.int16)
    v = bgr.max(axis=2)
    # Blue-dominant and not bright: sea. Specular flecks are bright but sit
    # inside the sea region, so they are kept here and rejected downstream.
    mask = ((b > r + 4) & (v < 170)).astype(np.uint8) * 255
    if horizon_y is not None:
        mask[: int(horizon_y) + 1, :] = 0
    # Close small holes (the flecks) so the sea reads as one region.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:                        # keep only the largest water body
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (lab == big).astype(np.uint8) * 255
    return mask


def dark_candidates(
    gray: np.ndarray,
    mask: np.ndarray,
    min_area: float = 2.0,
    max_area: float = 600.0,
    k: float = 2.5,
) -> list[Candidate]:
    """Small dark blobs sitting on water.

    The vessel reads as a handful of dark pixels against a mean sea level that
    varies across the frame, so the threshold is local rather than global.
    """
    g = gray.astype(np.float32)
    # Local sea brightness, at a scale well above the vessel's size.
    bg = cv2.GaussianBlur(g, (0, 0), 21)
    diff = bg - g                               # positive where darker than surround
    sea = mask > 0
    if sea.sum() < 500:
        return []
    sd = float(diff[sea].std()) or 1.0
    hits = ((diff > k * sd) & sea).astype(np.uint8)
    hits = cv2.morphologyEx(hits, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    n, lab, stats, cent = cv2.connectedComponentsWithStats(hits, 8)
    out: list[Candidate] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue
        cx, cy = float(cent[i][0]), float(cent[i][1])
        strength = float(diff[lab == i].mean() / 255.0)
        out.append(Candidate(cx, cy, area, strength))
    out.sort(key=lambda c: -c.contrast)
    return out


def analyse(bgr: np.ndarray, max_candidates: int = 40) -> Scene:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hy = find_horizon(gray)
    wm = water_mask(bgr, hy)
    cands = dark_candidates(gray, wm)[:max_candidates]
    return Scene(hy, wm, cands)
