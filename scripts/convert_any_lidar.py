#!/usr/bin/env python3
import argparse, pathlib, numpy as np, sys

p = argparse.ArgumentParser()
p.add_argument("--in_dir", required=True)
p.add_argument("--out_dir", required=True)
p.add_argument("--pattern", default="*")
p.add_argument("--unit_scale", type=float, default=1.0)
args = p.parse_args()

IN = pathlib.Path(args.in_dir)
OUT = pathlib.Path(args.out_dir)
OUT.mkdir(parents=True, exist_ok=True)

def load_points(path: pathlib.Path):
    s = path.suffix.lower()
    if s == ".npy":
        pts = np.load(path)
        return pts[:, :3] if pts.ndim == 2 else np.zeros((0, 3), np.float32)
    if s == ".bin":
        a = np.fromfile(path, np.float32)
        n = a.size // 4
        a = a[: n * 4].reshape(-1, 4)  # x,y,z,intensity
        return a[:, :3]
    try:
        if s in [".pcd", ".ply"]:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(str(path))
            return np.asarray(pcd.points, dtype=np.float32)
        if s in [".las", ".laz"]:
            import laspy
            las = laspy.read(str(path))
            return np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
    except Exception as e:
        print(f"[WARN] {path.name}: {e}", file=sys.stderr)
    return np.zeros((0, 3), np.float32)

cnt = 0
for f in IN.rglob(args.pattern):
    if not f.is_file():
        continue
    pts = load_points(f).astype(np.float32)
    if pts.size == 0:
        continue
    if args.unit_scale != 1.0:
        pts[:, :3] *= args.unit_scale
    out = OUT / (f.stem + ".npy")
    np.save(out, pts)
    cnt += 1

print(f"[DONE] wrote {cnt} npy files to {OUT}")
