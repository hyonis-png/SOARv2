"""Point tower-2 at water; fly the fixed-wing and quad in Z-patterns over the strait."""
from __future__ import annotations
import math, sys, time
sys.path.insert(0, ".")
import numpy as np
from soar import link, geo

YAW_OFF = 45.0                       # tracker reports grid north (EPSG:3413)
EE, NN, Z = np.load("data/site/grid.npy")
G, E0, N0 = 60.0, EE[0, 0], NN[0, 0]

def is_water(e, n):
    ci = int(np.clip(round((e - E0) / G), 0, EE.shape[1] - 1))
    ri = int(np.clip(round((n - N0) / G), 0, EE.shape[0] - 1))
    return Z[ri, ci] <= 0

def water_frac(oe, on, brg, far=1500.0, k=40):
    d = np.linspace(80, far, k); b = math.radians(brg)
    return float(np.mean([is_water(oe + x * math.sin(b), on + x * math.cos(b)) for x in d]))

def strait_axis():
    sea = Z <= 0
    pts = np.stack([EE[sea], NN[sea]], 1)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    ax = vt[0]
    if ax[0] < 0: ax = -ax
    t = (pts - c) @ ax
    return c, ax, t.min(), t.max()

def zigzag(n_legs=8, cross=600.0, alt=120.0):
    """Waypoints tracking the strait axis, alternating across it."""
    c, ax, lo, hi = strait_axis()
    perp = np.array([-ax[1], ax[0]])
    wps = []
    for i in range(n_legs + 1):
        s = lo + (hi - lo) * i / n_legs
        off = cross * (1 if i % 2 else -1)
        p = c + ax * s + perp * off
        if not is_water(*p):                       # pull back onto water
            for f in (0.5, 0.0, -0.5, -1.0):
                q = c + ax * s + perp * (cross * f)
                if is_water(*q): p = q; break
        lat, lon = geo.to_latlon(*p)
        wps.append((lat, lon, alt))
    return wps

def point_tower(name="tower-2"):
    a = link.Asset(name); assert a.connect(timeout=12)
    a.set_mode(link.TRACKER_MODES["MANUAL"]); time.sleep(0.8)
    t = a.snapshot(); oe, on = geo.to_enu(t.lat, t.lon)
    best = max(range(0, 360, 5), key=lambda b: water_frac(oe, on, b))
    print(f"{name}: best water bearing {best} deg (frac {water_frac(oe,on,best):.2f})")
    pwm = 1500
    for _ in range(8):
        a.set_servo(1, pwm); time.sleep(1.3)
        y = a.snapshot().yaw_deg
        if y is None: break
        err = ((best - (y + YAW_OFF) % 360 + 180) % 360) - 180
        print(f"   pwm {pwm} -> brg {(y+YAW_OFF)%360:6.1f}  err {err:+6.1f}")
        if abs(err) < 2.0: break
        pwm = int(round(max(1100, min(1900, pwm - err / 0.356))))
    a.close()

def fly(name, wps, alt, plane):
    a = link.Asset(name); assert a.connect(timeout=12)
    print(f"\n{name}: arming + takeoff to {alt} m")
    a.set_mode(link.PLANE_MODES["GUIDED"] if plane else link.COPTER_MODES["GUIDED"])
    time.sleep(0.5)
    a.arm(True)                       # guided window is ~3 s: no gap here
    if plane:
        a.set_mode(link.PLANE_MODES["TAKEOFF"])
    else:
        a.takeoff(alt)
    for _ in range(40):
        time.sleep(1.0)
        t = a.snapshot()
        if t.rel_alt_m and t.rel_alt_m > alt * 0.6: break
    print(f"   armed={a.snapshot().armed} alt={a.snapshot().rel_alt_m}")
    a.set_mode(link.PLANE_MODES["GUIDED"] if plane else link.COPTER_MODES["GUIDED"])
    return a


def run_legs(a, wps, accept=180.0, leg_timeout=180.0):
    """Fly the pattern, waiting for arrival before commanding the next leg.

    Sending the whole list at once just makes the vehicle chase the last point;
    GUIDED holds one target at a time.
    """
    for i, (lat, lon, z) in enumerate(wps):
        a.goto(lat, lon, z)
        tgt, t0 = geo.to_enu(lat, lon), time.time()
        while time.time() - t0 < leg_timeout:
            time.sleep(3.0)
            t = a.snapshot()
            if t.lat is None: continue
            d = math.dist(geo.to_enu(t.lat, t.lon), tgt)
            if d < accept:
                print(f"   {a.name} leg {i} reached ({d:.0f} m)", flush=True); break
            a.goto(lat, lon, z)          # refresh: GUIDED targets time out
        else:
            print(f"   {a.name} leg {i} timed out", flush=True)
    return a

if __name__ == "__main__":
    point_tower("tower-2")
    wp_plane = zigzag(8, 600.0, 120.0)
    wp_quad  = zigzag(5, 350.0, 80.0)
    import threading
    p = fly("fixed-wing", wp_plane, 120.0, True)
    q = fly("quadcopter", wp_quad, 80.0, False)
    ts = [threading.Thread(target=run_legs, args=(p, wp_plane), daemon=True),
          threading.Thread(target=run_legs, args=(q, wp_quad), daemon=True)]
    for t in ts: t.start()
    for t in ts: t.join()
    print("\npatterns complete")
