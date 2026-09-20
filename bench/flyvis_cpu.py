"""Gating measurement: can flyvis hold real time on this machine?

The whole fly-vision frontend rests on one number. The plan says that if 721
columns x 65 cell types cannot keep up with the tower dwell budget on 8 CPU
cores with no GPU, the frontend becomes a hand-rolled T4/T5 correlator -- and
that the decision gets made by a measurement rather than a preference.

What "real time" means here: the cameras publish at 10 fps, and a tower dwells
on one tile for a few seconds before saccading. So a dwell of D seconds carries
10*D frames, and the network must process them in under D seconds while the
tower is already slewing to the next tile. Anything under ~1.0x of wall-clock
per dwell is comfortable; over that and we fall behind the scan.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")

from flyvis import Network
from flyvis.datasets.rendering import BoxEye


def synthetic_roi(n_frames: int, size: int, seed: int = 0) -> np.ndarray:
    """A dark speck drifting across bright specular clutter.

    Stands in for a tower ROI: sea with glint, plus one small dark target
    translating coherently. Only the cost matters here, not the detection.
    """
    rng = np.random.default_rng(seed)
    vid = np.full((n_frames, size, size), 0.18, dtype=np.float32)
    # Specular flecks: bright, short-lived, stationary -- the false positives.
    for f in range(n_frames):
        ys = rng.integers(0, size, 120)
        xs = rng.integers(0, size, 120)
        vid[f, ys, xs] = rng.uniform(0.6, 1.0, 120)
    # The vessel: a few dark pixels, moving steadily.
    for f in range(n_frames):
        x = int(size * 0.2 + f * 0.35) % (size - 6)
        y = int(size * 0.55)
        vid[f, y:y + 2, x:x + 3] = 0.03
    return vid


def bench(dwell_s: float, cam_fps: float = 10.0, net_hz: float = 50.0,
          upsample: int = 4, repeats: int = 3) -> dict:
    """Time one dwell's worth of frames through the network.

    The cameras publish at 10 fps but the network was trained at dt=0.02, i.e.
    50 Hz; feeding it dt=0.1 raises an IntegrationWarning and puts the cell
    dynamics outside the regime they were fitted in. So the hexal timeseries is
    resampled 10 Hz -> 50 Hz before simulation.

    The resampling happens in HEX space, after the box filter, not in pixel
    space: interpolating 721 floats per step is negligible, whereas upsampling
    391x391 pixel frames fivefold would dominate the whole budget.
    """
    eye = BoxEye(extent=15, kernel_size=13)
    h, w = (int(v) for v in eye.min_frame_size)

    net = Network()
    net.eval()

    n_cam = max(2, int(round(dwell_s * cam_fps)))
    # A ~100 px ROI upsampled to the lattice's native frame size: this is the
    # "match the ommatidial acceptance angle to the target" step, so the vessel
    # subtends roughly one ommatidium instead of vanishing below one.
    roi_px = max(h, w) // upsample
    vid = synthetic_roi(n_cam, roi_px)

    t0 = time.perf_counter()
    frames = torch.from_numpy(vid)[:, None]                       # (T,1,H,W)
    frames = torch.nn.functional.interpolate(frames, size=(h, w), mode="bilinear",
                                             align_corners=False)
    hexed = eye(frames)                                           # (T,1,1,hexals)
    render_s = time.perf_counter() - t0

    # --- resample to the network's native rate, in hex space ---
    t0 = time.perf_counter()
    n_net = max(2, int(round(dwell_s * net_hz)))
    seq = hexed.permute(1, 2, 3, 0)                               # (1,1,hexals,T)
    seq = torch.nn.functional.interpolate(seq, size=(seq.shape[2], n_net),
                                          mode="bilinear", align_corners=True)
    movie = seq.permute(0, 3, 1, 2).contiguous()                  # (1,n_net,1,hexals)
    resample_s = time.perf_counter() - t0

    dt = 1.0 / net_hz
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = net.simulate(movie, dt=dt)
            times.append(time.perf_counter() - t0)

    sim_s = float(np.median(times))
    total_s = sim_s + render_s + resample_s
    return {
        "dwell_s": dwell_s,
        "cam_frames": n_cam,
        "net_steps": n_net,
        "roi_px": roi_px,
        "lattice": (h, w),
        "hexals": eye.hexals,
        "cells": tuple(out.shape),
        "render_s": render_s,
        "resample_s": resample_s,
        "sim_s": sim_s,
        "total_s": total_s,
        "realtime_factor": total_s / dwell_s,
    }


if __name__ == "__main__":
    torch.set_num_threads(8)
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | cpu only\n")

    hdr = (f"{'dwell':>6s} {'cam':>5s} {'steps':>6s} {'roi_px':>7s} "
           f"{'render':>8s} {'resamp':>8s} {'sim':>8s} {'total':>8s} {'xRT':>7s}  verdict")
    print(hdr); print("-" * len(hdr))

    worst = 0.0
    for dwell in (1.0, 1.5, 3.0, 5.0):
        r = bench(dwell)
        worst = max(worst, r["realtime_factor"])
        verdict = "ok" if r["realtime_factor"] < 1.0 else "TOO SLOW"
        print(f"{r['dwell_s']:6.1f} {r['cam_frames']:5d} {r['net_steps']:6d} {r['roi_px']:7d} "
              f"{r['render_s']:7.3f}s {r['resample_s']:7.3f}s {r['sim_s']:7.3f}s "
              f"{r['total_s']:7.3f}s {r['realtime_factor']:6.2f}x  {verdict}")

    print(f"\ncell types x columns in output: {r['cells']}")
    print(f"worst realtime factor: {worst:.2f}x")
    print("\nVERDICT:", "flyvis holds real time -> use it as the frontend"
          if worst < 1.0 else "flyvis too slow -> hand-rolled T4/T5 correlator")
