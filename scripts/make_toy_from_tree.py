#!/usr/bin/env python3
import argparse, os, re, sys, numpy as np
from pathlib import Path
from tqdm import tqdm

def ts_from_name(stem: str):
    m = re.findall(r"\d+", stem)
    return int(m[-1]) if m else None

def load_points(path: Path):
    suf = path.suffix.lower()
    try:
        if suf == ".bin":
            arr = np.fromfile(str(path), dtype=np.float32)
            if arr.size % 4 == 0:
                pts = arr.reshape(-1, 4)[:, :3]
            else:
                pts = arr.reshape(-1, 3)
            return pts.astype(np.float32)

        if suf == ".pcd":
            try:
                import open3d as o3d
            except Exception:
                return None
            pc = o3d.io.read_point_cloud(str(path))
            return np.asarray(pc.points, dtype=np.float32)

        if suf in {".npy"}:
            obj = np.load(str(path), allow_pickle=True)
            if isinstance(obj, np.ndarray):
                arr = obj
            else:
                return None
            # expect Nx3 or Nx>=3
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return arr[:, :3].astype(np.float32)
            return None

        if suf in {".npz"}:
            z = np.load(str(path))
            for key in ["points", "xyz", "lidar", "data", "arr_0"]:
                if key in z:
                    arr = z[key]
                    if arr.ndim == 2 and arr.shape[1] >= 3:
                        return arr[:, :3].astype(np.float32)
            return None

        if suf in {".ply"}:
            try:
                import open3d as o3d
            except Exception:
                return None
            pc = o3d.io.read_point_cloud(str(path))
            return np.asarray(pc.points, dtype=np.float32)

        if suf in {".las"}:
            try:
                import laspy
            except Exception:
                return None
            with laspy.open(str(path)) as f:
                pts = np.vstack([f.x, f.y, f.z]).T
                return pts.astype(np.float32)

        if suf in {".txt", ".csv"}:
            try:
                arr = np.loadtxt(str(path), dtype=np.float32, delimiter=None)
            except Exception:
                try:
                    arr = np.genfromtxt(str(path), dtype=np.float32, delimiter=",")
                except Exception:
                    return None
            if arr.ndim == 1:
                return None
            if arr.shape[1] >= 3:
                return arr[:, :3].astype(np.float32)
            return None

    except Exception:
        return None

    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--toy_name", default="nuscenes_toy")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth image")
    ap.add_argument("--max_pairs", type=int, default=10000)
    ap.add_argument("--img_exts", default=".jpg,.jpeg,.png,.bmp,.tif,.tiff")
    ap.add_argument("--lid_exts", default=".pcd,.bin,.npy,.npz,.ply,.las,.txt,.csv")
    ap.add_argument("--dry_run", action="store_true", help="only print what would be done")
    args = ap.parse_args()

    src = Path(args.src_root).expanduser()
    out = Path(args.out_root).expanduser()/args.toy_name
    img_dir = out/"images"; lid_dir = out/"lidar"
    img_dir.mkdir(parents=True, exist_ok=True); lid_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {e.strip().lower() for e in args.img_exts.split(",")}
    lid_exts = {e.strip().lower() for e in args.lid_exts.split(",")}

    # discover
    images = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in img_exts]
    lidars = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in lid_exts]

    print(f"[scan] found {len(images)} images and {len(lidars)} lidar files under {src}")
    if images[:5]:
        print("  examples (images):")
        for p in images[:5]: print("   -", p)
    if lidars[:5]:
        print("  examples (lidar):")
        for p in lidars[:5]: print("   -", p)

    if not images or not lidars:
        print("[!] No images or LiDAR found under", src)
        sys.exit(1)

    images.sort()
    lidars.sort()
    lid_ts = np.array([ts_from_name(p.stem) if ts_from_name(p.stem) is not None else -1 for p in lidars])

    if args.dry_run:
        print("[dry_run] stopping before writing any files.")
        return

    # pair & write
    written = 0
    for i, im in enumerate(tqdm(images[::args.stride], desc="Pairing & writing")):
        if written >= args.max_pairs:
            break
        tsim = ts_from_name(im.stem)
        if tsim is not None and (lid_ts >= 0).any():
            j = int(np.argmin(np.abs(lid_ts - tsim)))
        else:
            j = written % len(lidars)
        lp = lidars[j]
        pts = load_points(lp)
        if pts is None or pts.size == 0:
            continue

        out_im = img_dir/f"{im.stem}.jpg"
        # write image (copy or re-encode)
        try:
            import cv2
            img = cv2.imread(str(im))
            if img is None:
                raise RuntimeError("cv2 read failed")
            cv2.imwrite(str(out_im), img)
        except Exception:
            from shutil import copyfile
            try:
                copyfile(str(im), str(out_im))
            except Exception:
                continue

        np.save(lid_dir/f"{im.stem}.npy", pts.astype(np.float32))
        written += 1

    print(f"Done. Wrote {written} pairs to {img_dir} and {lid_dir}")

if __name__ == "__main__":
    main()
