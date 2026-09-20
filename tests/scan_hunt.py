"""Saccade-and-fixate across a tower's water arc, recording every frame.

This is the scan pattern the mission will use, run here to harvest footage that
actually contains the vessel. Dwell is deliberately long: at 10 fps a 3 m/s
target at 1 km moves only ~0.33 px/frame, so short dwells show no motion at all.
"""
from __future__ import annotations
import sys, os, json, time
sys.path.insert(0, ".")
import cv2
from soar import link, video

name   = sys.argv[1] if len(sys.argv) > 1 else "tower-1"
outdir = sys.argv[2] if len(sys.argv) > 2 else "data/clips/hunt_t1"
dwell  = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
passes = int(sys.argv[4]) if len(sys.argv) > 4 else 2
tiles  = [int(x) for x in (sys.argv[5].split(",") if len(sys.argv) > 5
                           else "1480,1620,1760,1890".split(","))]
os.makedirs(outdir, exist_ok=True)

a = link.Asset(name); assert a.connect(timeout=10)
a.set_mode(link.TRACKER_MODES["MANUAL"]); time.sleep(0.8)
g = video.Grabber(name).start(); time.sleep(2.0)

idx = open(os.path.join(outdir, "index.jsonl"), "w")
n = 0
for p in range(passes):
    for pwm in tiles:
        a.set_servo(1, pwm)
        time.sleep(1.8)                 # slew, then let it settle dead still
        g.latest()
        t0 = time.time()
        got = 0
        while time.time() - t0 < dwell:
            f = g.wait(timeout=2.0)
            if f is None:
                continue
            t = a.snapshot()
            fn = f"{n:05d}.jpg"
            cv2.imwrite(os.path.join(outdir, fn), f.image)
            idx.write(json.dumps({"file": fn, "t": f.t, "pass": p, "pwm": pwm,
                                  "yaw": t.yaw_deg, "pitch": t.pitch_deg,
                                  "alt_m": t.alt_m}) + "\n")
            n += 1; got += 1
        print(f"pass {p} pwm {pwm} yaw {a.snapshot().yaw_deg:6.1f}: {got} frames", flush=True)

idx.close(); a.set_servo(1, 1500); time.sleep(1.0); g.stop(); a.close()
print(f"\n{n} frames -> {outdir}")
