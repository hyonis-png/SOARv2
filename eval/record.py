"""Record raw camera frames plus synchronised telemetry, for offline detector work.

Detection tuning needs footage we can replay deterministically. Frames are kept
as JPEG exactly as they came off the wire (no recompression), alongside a JSONL
index carrying the pointing solution for each frame so geometry can be
reconstructed later.
"""
from __future__ import annotations
import json, os, sys, time
sys.path.insert(0, ".")
from soar import link, video


def record(asset: str, seconds: float, outdir: str) -> dict:
    os.makedirs(outdir, exist_ok=True)
    a = link.Asset(asset)
    have_telem = a.connect(timeout=10.0)
    idx_path = os.path.join(outdir, "index.jsonl")
    n, t0 = 0, time.time()
    with open(idx_path, "w") as idx:
        for raw in video.iter_jpegs(asset):
            fn = f"{n:05d}.jpg"
            with open(os.path.join(outdir, fn), "wb") as fh:
                fh.write(raw)
            t = a.snapshot() if have_telem else None
            idx.write(json.dumps({
                "file": fn, "t": time.time(), "seq": n,
                "lat": t.lat if t else None, "lon": t.lon if t else None,
                "alt_m": t.alt_m if t else None, "rel_alt_m": t.rel_alt_m if t else None,
                "yaw_deg": t.yaw_deg if t else None, "pitch_deg": t.pitch_deg if t else None,
                "servo1": t.servo.get(1) if t else None,
                "servo2": t.servo.get(2) if t else None,
            }) + "\n")
            n += 1
            if time.time() - t0 >= seconds:
                break
    if have_telem:
        a.close()
    return {"asset": asset, "frames": n, "dir": outdir,
            "secs": round(time.time() - t0, 1), "telemetry": have_telem}


if __name__ == "__main__":
    asset = sys.argv[1] if len(sys.argv) > 1 else "tower-1"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    out = sys.argv[3] if len(sys.argv) > 3 else f"data/clips/{asset}"
    print(json.dumps(record(asset, secs, out), indent=2))
