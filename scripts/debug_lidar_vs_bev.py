import argparse, numpy as np
from pathlib import Path
from PIL import Image
import torch
from src.models.simplebev_xs import lidar_hist_bev, _heat_to_img

ap = argparse.ArgumentParser()
ap.add_argument("lidar_npy_or_bin")
ap.add_argument("--out", default="/tmp/debug_bev.png")
ap.add_argument("--H", type=int, default=128)
ap.add_argument("--W", type=int, default=128)
ap.add_argument("--meters_x", type=float, default=80)
ap.add_argument("--meters_y", type=float, default=60)
ap.add_argument("--unit_scale", type=float, default=0.01)
ap.add_argument("--z_min", type=float, default=-3)
ap.add_argument("--z_max", type=float, default=3)
ap.add_argument("--fov_deg", type=float, default=360)
ap.add_argument("--swap_xy", action="store_true")
ap.add_argument("--flip_x", action="store_true")
ap.add_argument("--flip_y", action="store_true")
args = ap.parse_args()

p = Path(args.lidar_npy_or_bin)
if p.suffix == ".npy":
    pts = np.load(p).astype(np.float32)
else:
    # assume float32 XYZ in .bin; adjust if your format differs
    pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)[:, :3]

if args.unit_scale != 1.0:
    pts[:, :3] *= args.unit_scale
if args.swap_xy:
    pts[:, [0, 1]] = pts[:, [1, 0]]
if args.flip_x:
    pts[:, 0] = -pts[:, 0]
if args.flip_y:
    pts[:, 1] = -pts[:, 1]

bev = lidar_hist_bev(
    pts,
    bev_hw=(args.H, args.W),
    meters=(args.meters_x, args.meters_y),
    z_range=(args.z_min, args.z_max),
    fov_deg=args.fov_deg,
    blur_ksize=3, dilate_ks=5, log_scale=True
).squeeze().numpy()

print(f"raw pts: {pts.shape}, finite: {np.isfinite(pts).all()}")
print(f"bev: shape={bev.shape}, min={bev.min():.4f}, max={bev.max():.4f}, mean={bev.mean():.4f}")

# visualize with stretch to expose tiny signals
img = _heat_to_img(bev, p_low=0.5, p_high=99.5, gamma=0.6, use_colormap=True)
img.save(args.out)
print("Wrote", args.out)

