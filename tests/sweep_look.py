"""Sweep a tower across its pan range, grabbing one frame per bearing."""
from __future__ import annotations
import sys, os, time, json
sys.path.insert(0, ".")
import cv2, numpy as np
from soar import link, video

name = sys.argv[1] if len(sys.argv) > 1 else "tower-1"
outdir = sys.argv[2] if len(sys.argv) > 2 else "data/clips/sweep_t1"
os.makedirs(outdir, exist_ok=True)

a = link.Asset(name); assert a.connect(timeout=10)
a.set_mode(link.TRACKER_MODES["MANUAL"]); time.sleep(0.8)
g = video.Grabber(name).start()
time.sleep(2.0)

rows = []
pwms = list(range(1100, 1901, 50))
for pwm in pwms:
    a.set_servo(1, pwm)
    time.sleep(1.6)                       # slew + settle
    g.latest()                            # discard anything captured mid-slew
    time.sleep(0.4)
    f = g.wait(timeout=4.0)
    if f is None:
        print(f"pwm {pwm}: no frame"); continue
    t = a.snapshot()
    fn = f"{pwm}.jpg"
    cv2.imwrite(os.path.join(outdir, fn), f.image)
    # How much of the frame is dark, low-saturation sea rather than sky or land?
    hsv = cv2.cvtColor(f.image, cv2.COLOR_BGR2HSV)
    dark = (hsv[:, :, 2] < 150)
    rows.append({"pwm": pwm, "yaw": t.yaw_deg, "file": fn, "dark_frac": float(dark.mean())})
    print(f"pwm {pwm:5d}  yaw {t.yaw_deg:6.1f}  dark {dark.mean()*100:5.1f}%")

a.set_servo(1, 1500); time.sleep(1.5); g.stop(); a.close()
json.dump(rows, open(os.path.join(outdir, "sweep.json"), "w"), indent=2)

# Contact sheet so the whole arc can be eyeballed at once.
tiles = []
for r in rows:
    im = cv2.imread(os.path.join(outdir, r["file"]))
    im = cv2.resize(im, (320, 180))
    cv2.putText(im, f"{r['yaw']:.0f}deg", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    tiles.append(im)
cols = 4
while len(tiles) % cols:
    tiles.append(np.zeros((180, 320, 3), np.uint8))
sheet = np.vstack([np.hstack(tiles[i:i+cols]) for i in range(0, len(tiles), cols)])
cv2.imwrite(os.path.join(outdir, "contact.jpg"), sheet)
print(f"\ncontact sheet: {outdir}/contact.jpg  ({sheet.shape[1]}x{sheet.shape[0]})")
