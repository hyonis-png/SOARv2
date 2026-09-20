"""Live mission: scan tower-1's water arc, detect, geolocate, report.

Bearing note: the tracker reports GRID north (EPSG:3413), not true north. The
offset solved against the DEM is +45 deg, corr 0.976 -- consistent with the
site's -49.8 deg grid convergence.
"""
from __future__ import annotations
import math, sys, time
sys.path.insert(0, ".")
import numpy as np
from soar import link, video, geo, report
from soar.detect import baseline

YAW_OFFSET = 45.0
TRACK_NAME = "Sierra Two"
# The tracker's yaw reference drifts between runs -- the same reported yaw
# showed water one hour and bare land the next -- so the SEARCH sweeps the pan
# range mechanically and lets the detector decide what is water. Yaw is still
# read per-dwell for geolocation, but never trusted to aim.
SWEEP_PWM = list(range(1100, 1901, 100))
DWELL = 8.0

def settle(a, timeout=5.0, tol=0.25):
    last, stable, t0 = None, 0, time.time()
    while time.time() - t0 < timeout:
        y = a.snapshot().yaw_deg
        if y is not None and last is not None and abs(((y-last+180) % 360)-180) < tol:
            stable += 1
            if stable >= 3: return y
        else: stable = 0
        last = y; time.sleep(0.15)
    return a.snapshot().yaw_deg

def point_at(a, target_true, tries=7, tol=1.5):
    """Drive the pan servo until reported bearing matches, closing the loop.

    Open-loop PWM drifts: the same command gave 208.6 deg on one pass and 228.7
    on the next, so the commanded value cannot be trusted and we correct against
    what the tracker actually reports.
    """
    pwm = 1500
    for _ in range(tries):
        a.set_servo(1, pwm); time.sleep(1.1)
        y = settle(a, timeout=3.0)
        if y is None: return None, pwm
        cur = (y + YAW_OFFSET) % 360
        err = ((target_true - cur + 180) % 360) - 180
        if abs(err) < tol: return cur, pwm
        pwm = int(round(max(1100, min(1900, pwm - err / 0.356))))
    return cur, pwm


def run(name="tower-1", passes=1):
    cam = geo.CAMERAS[name]
    a = link.Asset(name)
    if not a.connect(timeout=12): sys.exit(f"{name} did not answer")
    a.set_mode(link.TRACKER_MODES["MANUAL"]); time.sleep(0.8)
    g = video.Grabber(name).start(); time.sleep(2.0)
    origin = None; fixes = []
    for p in range(passes):
        for pwm in SWEEP_PWM:
            a.set_servo(1, pwm); time.sleep(1.4); settle(a, timeout=3.0)
            g.latest()
            frames, t0 = [], time.time()
            while time.time() - t0 < DWELL:
                f = g.wait(timeout=2.0)
                if f is not None: frames.append(f.image)
            t = a.snapshot()
            if origin is None and t.lat: origin = geo.to_enu(t.lat, t.lon)
            hits = baseline.detect(frames)
            brg_c = (t.yaw_deg + YAW_OFFSET) % 360
            print(f"pwm {pwm} brg~{brg_c:5.1f}  {len(frames):3d} frames  "
                  f"{len(hits)} hit(s)", flush=True)
            for h in hits[:1]:
                rng = geo.range_from_horizon(cam, t.alt_m, h.y, t.pitch_deg)
                if rng is None or rng > geo.FAR_CLIP_M:
                    print(f"    reject: range={rng}"); continue
                brg = (brg_c + cam.yaw_offset_deg(h.x)) % 360
                e, n = geo.project(origin, rng, brg)
                lat, lon = geo.to_latlon(e, n)
                print(f"    CONTACT area={h.area:.0f} rng={rng:.0f}m brg={brg:.1f} "
                      f"-> {lat:.6f},{lon:.6f}")
                fixes.append((time.time(), lat, lon))
                hd = sp = None
                if len(fixes) >= 2:
                    (t0_, la0, lo0), (t1_, la1, lo1) = fixes[-2], fixes[-1]
                    p0, p1 = geo.to_enu(la0, lo0), geo.to_enu(la1, lo1)
                    d, hd = geo.range_bearing(p0, p1)
                    sp = d / max(1e-3, t1_ - t0_)
                res = report.post_track(TRACK_NAME, lat, lon, hd, sp)
                print(f"    POSTED -> uuid={res.get('uuid')} created={res.get('created')} "
                      f"fixes={res.get('fixes')}")
    a.set_servo(1, 1500); g.stop(); a.close()
    return fixes

if __name__ == "__main__":
    fx = run(passes=int(sys.argv[1]) if len(sys.argv) > 1 else 1)
    print(f"\n{len(fx)} fix(es) reported")
    for t in report.list_tracks().get("tracks", []):
        if t["name"] == TRACK_NAME:
            print(f"server: {t['name']} {t['lat']:.6f},{t['lon']:.6f} fixes={t['fixes']}")
