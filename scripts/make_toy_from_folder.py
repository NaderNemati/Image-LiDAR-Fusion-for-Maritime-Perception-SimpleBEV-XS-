#!/usr/bin/env python3
import argparse, os, re, sys, numpy as np
from pathlib import Path
from tqdm import tqdm

def ts_from_name(stem: str):
    m = re.findall(r"\d+", stem)
    return int(m[-1]) if m else None

def load_points(path: Path):
    if path.suffix.lower() == ".bin":
        arr = np.fromfile(str(path), dtype=np.float32)
        n = arr.size // 4
        return arr.reshape(n,4)[:, :3]
    elif path.suffix.lower() == ".pcd":
        try:
            import open3d as o3d
        except Exception:
            return None
        pc = o3d.io.read_point_cloud(str(path))
        return np.asarray(pc.points, dtype=np.float32)
    else:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--toy_name", default="nuscenes_toy")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth image")
    ap.add_argument("--max_pairs", type=int, default=10_000)
    ap.add_argument("--img_exts", default=".jpg,.jpeg,.png")
    ap.add_argument("--lid_exts", default=".pcd,.bin")
    args = ap.parse_args()

    src = Path(args.src_root).expanduser()
    out = Path(args.out_root).expanduser()/args.toy_name
    img_dir = out/"images"; lid_dir = out/"lidar"
    img_dir.mkdir(parents=True, exist_ok=True); lid_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {e.strip().lower() for e in args.img_exts.split(",")}
    lid_exts = {e.strip().lower() for e in args.lid_exts.split(",")}

    images = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in img_exts]
    lidars = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in lid_exts]
    if not images or not lidars:
        print("[!] No images or LiDAR found under", src); sys.exit(1)

    images.sort()
    lidars.sort()
    lid_ts = np.array([ts_from_name(p.stem) if ts_from_name(p.stem) is not None else -1 for p in lidars])

    written = 0
    for i, im in enumerate(tqdm(images[::args.stride], desc="Pairing & writing")):
        if written >= args.max_pairs: break
        tsim = ts_from_name(im.stem)
        # nearest timestamp pairing, fallback to cycling index
        if tsim is not None and (lid_ts >= 0).any():
            j = int(np.argmin(np.abs(lid_ts - tsim)))
        else:
            j = written % len(lidars)
        lp = lidars[j]
        pts = load_points(lp)
        if pts is None or pts.size == 0: 
            continue

        # write image
        out_im = img_dir/f"{im.stem}.jpg"
        try:
            import cv2
            img = cv2.imread(str(im))
            if img is None: 
                continue
            cv2.imwrite(str(out_im), img)
        except Exception:
            # fallback copy
            from shutil import copyfile
            try: copyfile(str(im), str(out_im))
            except Exception: continue

        np.save(lid_dir/f"{im.stem}.npy", pts.astype(np.float32))
        written += 1

    print(f"Done. Wrote {written} pairs to {img_dir} and {lid_dir}")

if __name__ == "__main__":
    main()

