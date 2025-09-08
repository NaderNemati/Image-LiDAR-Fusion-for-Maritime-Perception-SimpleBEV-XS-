#!/usr/bin/env python3
"""
Make (image, lidar) pairs by time with optional offset/auto-scan.
Writes CSV: image_path,lidar_bin_path,dt_ms
Optional: write LiDAR .npy named by IMAGE STEM into --npy_dir.
"""

import argparse, time, json, math
from pathlib import Path
from PIL import Image, ExifTags
import numpy as np
import bisect
from typing import List, Tuple, Optional

# ---------- helpers ----------
def is_numeric_stem(stem: str) -> bool:
    return stem.isdigit()

def parse_epoch_ms_from_stem(stem: str) -> Optional[int]:
    """Accept 13 (ms), 16 (us), 19 (ns) digits; return ms."""
    if not stem.isdigit():
        return None
    n = len(stem)
    v = int(stem)
    if n >= 19:       # ns
        return v // 1_000_000
    elif n >= 16:     # us
        return v // 1_000
    elif n >= 13:     # ms
        return v
    else:             # seconds
        return v * 1000

_EXIF_DT = next((k for k, v in ExifTags.TAGS.items() if v == "DateTimeOriginal"), None)

def image_time_ms(p: Path) -> int:
    """Prefer EXIF DateTimeOriginal (local time), fallback file mtime."""
    ms = None
    try:
        ex = Image.open(p).getexif()
        if ex and _EXIF_DT in ex:
            s = ex[_EXIF_DT]  # "YYYY:MM:DD HH:MM:SS" (local time)
            t = time.strptime(s, "%Y:%m:%d %H:%M:%S")
            ms = int(time.mktime(t) * 1000)
    except Exception:
        pass
    if ms is None:
        ms = int(p.stat().st_mtime * 1000)
    return ms

def lidar_time_ms(p: Path) -> Optional[int]:
    return parse_epoch_ms_from_stem(p.stem)

def utc_str(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(ms/1000))

def nearest_idx(a: List[int], x: int) -> int:
    i = bisect.bisect_left(a, x)
    if i == 0: return 0
    if i == len(a): return len(a) - 1
    return i if abs(a[i] - x) < abs(a[i-1] - x) else i-1

# ---------- core ----------
def auto_offset_ms(img_ms: List[int], lid_ms: List[int], scan_hours: int, step_ms: int = 60_000) -> Tuple[int, int]:
    """
    Coarse search over offsets (apply to images) in ±scan_hours to maximize matches within 60s window.
    Returns (best_offset_ms, best_matches).
    """
    if not img_ms or not lid_ms:
        return (0, 0)
    lid_set = lid_ms
    best = (0, -1)  # (offset, matches)
    window = 60_000
    rng = range(-scan_hours*3600_000, scan_hours*3600_000 + 1, step_ms)
    for off in rng:
        m = 0
        j_used = set()
        for t in img_ms:
            target = t + off
            j = nearest_idx(lid_set, target)
            if j not in j_used and abs(lid_set[j] - target) <= window:
                j_used.add(j)
                m += 1
        if m > best[1]:
            best = (off, m)
    return best

def pair_times(img_times: List[Tuple[int, Path]],
               lid_times: List[Tuple[int, Path]],
               img_offset_ms: int,
               max_ms: int) -> List[Tuple[Path, Path, int]]:
    """Greedy one-pass pairing without reusing LiDAR frames."""
    img_times = sorted(img_times, key=lambda x: x[0])
    lid_times = sorted(lid_times, key=lambda x: x[0])
    lids_only = [t for t, _ in lid_times]

    pairs = []
    used = set()
    for t_img, p_img in img_times:
        target = t_img + img_offset_ms
        j = nearest_idx(lids_only, target) if lids_only else None
        if j is None: break
        # search small neighborhood to reduce |dt|
        cand = []
        for k in (j-2, j-1, j, j+1, j+2):
            if 0 <= k < len(lid_times) and k not in used:
                dt = lid_times[k][0] - target
                if abs(dt) <= max_ms:
                    cand.append((abs(dt), k, dt))
        if not cand: 
            continue
        cand.sort()
        _, best_k, dt = cand[0]
        used.add(best_k)
        pairs.append((p_img, lid_times[best_k][1], int(dt)))
    return pairs

def write_npy_for_pairs(pairs, npy_dir: Path, unit_scale: float, swap_xy: bool, flip_x: bool, flip_y: bool):
    npy_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_p, bin_p, _ in pairs:
        try:
            arr = np.fromfile(str(bin_p), dtype=np.float32).reshape(-1, 4)  # (x,y,z,rest)
            pts = arr[:, :3].astype(np.float32)
            if unit_scale != 1.0:
                pts *= unit_scale
            if swap_xy:
                pts[:, [0, 1]] = pts[:, [1, 0]]
            if flip_x:
                pts[:, 0] = -pts[:, 0]
            if flip_y:
                pts[:, 1] = -pts[:, 1]
            outp = npy_dir / (img_p.stem + ".npy")
            np.save(outp, pts)
            n += 1
        except Exception as e:
            print(f"[WARN] failed npy for {bin_p.name}: {e}")
    return n

# ---------- cli ----------
def main():
    ap = argparse.ArgumentParser(description="Make (image, lidar) pairs by time with optional offset/auto-scan.")
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--lidar_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--max_ms", type=int, default=800, help="pairing tolerance")
    ap.add_argument("--img_offset_ms", type=int, default=0, help="add this offset to image times before pairing")
    ap.add_argument("--auto_offset", action="store_true", help="search for a good offset automatically")
    ap.add_argument("--scan_hours", type=int, default=12, help="±hours to search when --auto_offset is set")
    ap.add_argument("--write_npy", action="store_true", help="also write .npy files for LiDAR named by IMAGE stems")
    ap.add_argument("--npy_dir", default="datasets/nuscenes_toy/lidar")
    ap.add_argument("--unit_scale", type=float, default=1.0)
    ap.add_argument("--swap_xy", action="store_true")
    ap.add_argument("--flip_x", action="store_true")
    ap.add_argument("--flip_y", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    img_dir = Path(args.img_dir).expanduser()
    lid_dir = Path(args.lidar_dir).expanduser()
    out_csv = Path(args.out_csv)

    imgs = sorted([p for p in img_dir.glob("*.jpg")])
    lids = sorted([p for p in lid_dir.glob("*.bin")])

    img_times = []
    for p in imgs:
        t = parse_epoch_ms_from_stem(p.stem)
        if t is None:
            t = image_time_ms(p)
        img_times.append((t, p))
    lid_times = []
    for p in lids:
        t = lidar_time_ms(p)
        if t is not None:
            lid_times.append((t, p))

    it = sorted([t for t,_ in img_times])
    lt = sorted([t for t,_ in lid_times])

    print(f"[INFO] images={len(img_times)} (time-found={len(it)}), lidar_bins={len(lid_times)} (time-found={len(lt)})")
    if not it or not lt:
        print("[ERROR] Missing timestamps in one of the folders.")
        return

    # report windows
    print(f"[INFO] IMG  window: {utc_str(min(it))} → {utc_str(max(it))}")
    print(f"[INFO] LIDARwindow: {utc_str(min(lt))} → {utc_str(max(lt))}")

    offset = args.img_offset_ms
    if args.auto_offset:
        est, matched = auto_offset_ms(it, lt, args.scan_hours)
        print(f"[AUTO] best image-time offset ≈ {est} ms (~{est/3600000:.2f} h); matched≈{matched} within ±60s")
        offset = est

    pairs = pair_times(img_times, lid_times, offset, args.max_ms)
    print(f"[DONE] matched {len(pairs)} pairs (<= {args.max_ms} ms) with img_offset_ms={offset}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "lidar_bin", "dt_ms"])
        for ip, lp, dt in pairs:
            w.writerow([ip.as_posix(), lp.as_posix(), dt])
    print(f"[WROTE] {out_csv}")

    if args.write_npy and pairs:
        n = write_npy_for_pairs(pairs, Path(args.npy_dir), args.unit_scale, args.swap_xy, args.flip_x, args.flip_y)
        print(f"[WROTE] {n} npy files → {args.npy_dir} (named by IMAGE stems)")

if __name__ == "__main__":
    main()

