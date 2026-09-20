"""Establish the SERVO1 PWM -> yaw mapping for a tower, empirically.

The deck says `servo set 1 1900` pans and `servo set 2 1200` tilts, but not what
angle any PWM corresponds to. The scan behaviour needs that mapping, so measure
it rather than assume it. AntennaTracker reports its pan direction as the
vehicle's own yaw, so ATTITUDE is the readout.
"""
from __future__ import annotations
import sys, time, json
sys.path.insert(0, ".")
from soar import link

def settle(a: link.Asset, timeout=4.0, tol=0.3):
    """Wait until yaw stops changing."""
    last, stable, t0 = None, 0, time.time()
    while time.time() - t0 < timeout:
        y = a.snapshot().yaw_deg
        if y is not None and last is not None and abs(((y - last + 180) % 360) - 180) < tol:
            stable += 1
            if stable >= 3:
                return y
        else:
            stable = 0
        last = y
        time.sleep(0.15)
    return a.snapshot().yaw_deg

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "tower-1"
    a = link.Asset(name)
    assert a.connect(timeout=10), f"{name} did not answer"
    time.sleep(1.5)
    t = a.snapshot()
    print(f"{name}: mode={t.mode} armed={t.armed} yaw={t.yaw_deg:.1f} servo1={t.servo.get(1)}")

    a.set_mode(link.TRACKER_MODES["MANUAL"]); time.sleep(1.0)
    print(f"after MANUAL: mode={a.snapshot().mode}\n")

    rows = []
    print(f"{'pwm':>6s} {'yaw_deg':>8s} {'pitch':>7s} {'servo1':>7s} {'servo2':>7s}")
    for pwm in range(1100, 1901, 100):
        a.set_servo(1, pwm)
        y = settle(a)
        s = a.snapshot()
        rows.append({"pwm": pwm, "yaw": y, "pitch": s.pitch_deg,
                     "servo1": s.servo.get(1), "servo2": s.servo.get(2)})
        print(f"{pwm:6d} {y if y is not None else float('nan'):8.1f} "
              f"{s.pitch_deg if s.pitch_deg is not None else float('nan'):7.1f} "
              f"{s.servo.get(1, 0):7d} {s.servo.get(2, 0):7d}")

    a.set_servo(1, 1500); settle(a)
    a.close()
    json.dump(rows, open(f"data/pan_calib_{name}.json", "w"), indent=2)
    ys = [r["yaw"] for r in rows if r["yaw"] is not None]
    print(f"\nyaw span across PWM 1100..1900: {min(ys):.1f} .. {max(ys):.1f} "
          f"({max(ys)-min(ys):.1f} deg)")
