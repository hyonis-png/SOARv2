"""Verification step 2: heartbeat + ATTITUDE from all four assets over udpout."""
import sys, time
sys.path.insert(0, ".")
from soar import link

assets = link.connect_all(timeout=12.0)
print(f"connected: {sorted(assets)}  ({len(assets)}/4)\n")
time.sleep(4.0)  # let the reader threads fill in

hdr = f"{'asset':12s} {'mode':>5s} {'armed':>6s} {'lat':>11s} {'lon':>12s} {'alt':>8s} {'yaw':>7s} {'servo1':>7s} {'servo2':>7s}"
print(hdr); print("-" * len(hdr))
ok = 0
for name in sorted(assets):
    t = assets[name].snapshot()
    if t.lat is not None and t.yaw_deg is not None:
        ok += 1
    print(f"{name:12s} {str(t.mode):>5s} {str(t.armed):>6s} "
          f"{t.lat if t.lat is not None else float('nan'):11.6f} "
          f"{t.lon if t.lon is not None else float('nan'):12.6f} "
          f"{t.rel_alt_m if t.rel_alt_m is not None else float('nan'):8.1f} "
          f"{t.yaw_deg if t.yaw_deg is not None else float('nan'):7.1f} "
          f"{t.servo.get(1, 0):7d} {t.servo.get(2, 0):7d}")

for a in assets.values():
    a.close()
print(f"\n{ok}/4 assets reporting position and attitude")
sys.exit(0 if ok == 4 else 1)
