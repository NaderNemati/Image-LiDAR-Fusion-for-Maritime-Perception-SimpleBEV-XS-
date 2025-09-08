#!/usr/bin/env python3
import argparse, os, json, math
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# use your project rasterizer for consistency
from src.models.simplebev_xs import lidar_hist_bev, _norm01

def _load_points(path):
    p = Path(path)
    if p.suffix == ".bin":
        arr = np.fromfile(str(p), dtype=np.float32)
        if arr.size % 4 != 0:
            raise ValueError(f"{p} size not divisible by 4; got {arr.size}")
        pts = arr.reshape(-1, 4)[:, :3]
    elif p.suffix == ".npy":
        arr = np.load(str(p))
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"{p} must be Nx3/4, got {arr.shape}")
        pts = arr[:, :3].astype(np.float32)
    elif p.suffix == ".npz":
        data = np.load(str(p))
        # try common keys
        for k in ("points", "lidar", "xyz"):
            if k in data:
                pts = np.asarray(data[k], dtype=np.float32)[:, :3]
                break
        else:
            raise ValueError(f"{p} .npz has no 'points'/'lidar'/'xyz' key")
    else:
        raise ValueError(f"Unsupported extension: {p.suffix}")
    return pts

def _apply_units_axes_filters(pts, unit_scale, swap_xy, flip_x, flip_y,
                              z_min, z_max, fov_deg, meters_x, meters_y):
    out = pts.copy().astype(np.float32)
    if unit_scale != 1.0:
        out[:, :3] *= float(unit_scale)
    if swap_xy:
        out[:, [0, 1]] = out[:, [1, 0]]
    if flip_x:
        out[:, 0] = -out[:, 0]
    if flip_y:
        out[:, 1] = -out[:, 1]

    n0 = len(out)
    # Z filter
    m = (out[:,2] >= z_min) & (out[:,2] <= z_max)
    out = out[m]; z_drop = n0 - len(out)

    # FOV (angle in xy plane; +x forward)
    if fov_deg is not None and fov_deg < 360:
        ang = np.degrees(np.arctan2(out[:,1], out[:,0]))
        half = fov_deg * 0.5
        m = (ang >= -half) & (ang <= half)
        out = out[m]

    # metric window
    xr, yr = float(meters_x), float(meters_y)
    x1, x2 = -1e-6, xr
    y1, y2 = -yr/2.0, yr/2.0
    m = (out[:,0] >= x1) & (out[:,0] <= x2) & (out[:,1] >= y1) & (out[:,1] <= y2)
    final = out[m]
    return final

def _percentile_img(a, p_low, p_high, gamma=1.0):
    lo = np.percentile(a, p_low)
    hi = np.percentile(a, p_high)
    y  = (a - lo) / (hi - lo + 1e-6)
    y  = np.clip(y, 0, 1)
    if abs(gamma - 1.0) > 1e-6:
        y = np.power(y, 1.0/gamma)
    return (y * 255).astype(np.uint8)

def _project_points(pts, img_hw, K, T_cam_from_ego):
    H, W = img_hw
    P = np.concatenate([pts, np.ones((len(pts),1), np.float32)], axis=1)    # [N,4]
    Pc = (T_cam_from_ego @ P.T).T                                           # [N,4]
    z = Pc[:,2]
    valid = z > 1e-3
    Pc = Pc[valid]
    uv = (K @ Pc[:,:3].T).T
    uv = uv[:, :2] / (uv[:, 2:3] + 1e-6)
    m = (uv[:,0] >= 0) & (uv[:,0] < W) & (uv[:,1] >= 0) & (uv[:,1] < H)
    return uv[m]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lidar", help=".bin/.npy/.npz path")
    ap.add_argument("--out_dir", default="outputs/lidar_debug")
    ap.add_argument("--image", default="", help="optional RGB image path to show next to BEV")
    ap.add_argument("--calib", default="", help="optional calib.json for point->RGB projection")
    # raster window (match training)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--meters_x", type=float, default=60.0)
    ap.add_argument("--meters_y", type=float, default=30.0)
    ap.add_argument("--unit_scale", type=float, default=1.0)
    ap.add_argument("--z_min", type=float, default=-3.0)
    ap.add_argument("--z_max", type=float, default=3.0)
    ap.add_argument("--fov_deg", type=float, default=360.0)
    ap.add_argument("--swap_xy", action="store_true")
    ap.add_argument("--flip_x", action="store_true")
    ap.add_argument("--flip_y", action="store_true")
    # viz tuning
    ap.add_argument("--p_low", type=float, default=1.0)
    ap.add_argument("--p_high", type=float, default=99.0)
    ap.add_argument("--gamma", type=float, default=0.6)
    args = ap.parse_args()

    outdir = Path(args.out_dir); outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.lidar).stem

    # ----- load raw
    raw = _load_points(args.lidar)
    n_raw = len(raw)

    # ----- apply transforms/filters
    filt = _apply_units_axes_filters(raw, args.unit_scale, args.swap_xy, args.flip_x, args.flip_y,
                                     args.z_min, args_z_max:=args.z_max, args.fov_deg,
                                     args.meters_x, args.meters_y)
    n_filt = len(filt)

    # ----- make BEV with your rasterizer (matched settings)
    bev = lidar_hist_bev(
        filt,
        bev_hw=(args.H, args.W),
        meters=(args.meters_x, args.meters_y),
        z_range=(args.z_min, args_z_max),
        fov_deg=args.fov_deg,
        blur_ksize=0,
        dilate_ks=0,
        log_scale=True
    ).squeeze().numpy()

    nz = int((bev > 0).sum())
    stats = {
        "raw_points": int(n_raw),
        "after_filters": int(n_filt),
        "bev_shape": [int(args.H), int(args.W)],
        "bev_nonzero": nz,
        "bev_min": float(bev.min() if bev.size else 0.0),
        "bev_max": float(bev.max() if bev.size else 0.0),
        "bev_mean": float(bev.mean() if bev.size else 0.0),
        "meters_x": args.meters_x, "meters_y": args.meters_y,
        "z_min": args.z_min, "z_max": args_z_max, "fov_deg": args.fov_deg,
        "unit_scale": args.unit_scale, "swap_xy": bool(args.swap_xy),
        "flip_x": bool(args.flip_x), "flip_y": bool(args.flip_y)
    }
    print("[LiDAR DEBUG]", json.dumps(stats, indent=2))

    # ----- top view scatter
    plt.figure()
    if len(filt) > 0:
        plt.scatter(filt[:,0], filt[:,1], s=0.2)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(0, args.meters_x); plt.ylim(-args.meters_y/2.0, args.meters_y/2.0)
    plt.xlabel("x forward (m)"); plt.ylabel("y left (m)"); plt.title("Top View (raw points in window)")
    plt.tight_layout()
    plt.savefig(outdir / f"{stem}_topview.png", dpi=200)
    plt.close()

    # ----- side view scatter (x vs z)
    plt.figure()
    if len(filt) > 0:
        plt.scatter(filt[:,0], filt[:,2], s=0.2)
    plt.xlim(0, args.meters_x); plt.ylim(args.z_min, args_z_max)
    plt.xlabel("x forward (m)"); plt.ylabel("z up (m)"); plt.title("Side View (x–z)")
    plt.tight_layout()
    plt.savefig(outdir / f"{stem}_sideview.png", dpi=200)
    plt.close()

    # ----- histograms
    if len(filt) > 0:
        rng = np.linalg.norm(filt[:, :2], axis=1)
    else:
        rng = np.zeros((0,), np.float32)

    plt.figure()
    plt.hist(rng, bins=60)
    plt.xlabel("range (m)"); plt.ylabel("count"); plt.title("Range histogram")
    plt.tight_layout()
    plt.savefig(outdir / f"{stem}_range_hist.png", dpi=200)
    plt.close()

    plt.figure()
    if len(filt) > 0:
        plt.hist(filt[:,2], bins=60)
    else:
        plt.hist([], bins=60)
    plt.xlabel("z (m)"); plt.ylabel("count"); plt.title("Z histogram")
    plt.tight_layout()
    plt.savefig(outdir / f"{stem}_z_hist.png", dpi=200)
    plt.close()

    # ----- BEV heatmap (percentile stretch)
    bev_vis = _percentile_img(bev, args.p_low, args.p_high, gamma=args.gamma)
    Image.fromarray(bev_vis, mode="L").save(outdir / f"{stem}_bev.png")

    # ----- optional: show next to RGB
    if args.image and Path(args.image).exists():
        im = Image.open(args.image).convert("RGB")
        canvas_w = im.width + bev_vis.shape[1]
        canvas_h = max(im.height, bev_vis.shape[0])
        canv = Image.new("RGB", (canvas_w, canvas_h), (0,0,0))
        canv.paste(im, (0, 0))
        bev_rgb = Image.fromarray(bev_vis, mode="L").convert("RGB")
        bev_rgb = bev_rgb.resize((im.width, im.height))
        canv.paste(bev_rgb, (im.width, 0))
        canv.save(outdir / f"{stem}_rgb_plus_bev.png")

    # ----- optional: project onto RGB (needs calib)
    if args.calib and args.image and Path(args.calib).exists():
        data = json.loads(Path(args.calib).read_text())
        K  = np.array(data["cameras"][0]["K"], dtype=np.float32)
        T  = np.array(data["cameras"][0]["T_cam_from_ego"], dtype=np.float32)
        im = Image.open(args.image).convert("RGB")
        uv = _project_points(filt, (im.height, im.width), K, T)
        import cv2
        rgb = np.array(im).copy()
        uv_int = np.round(uv).astype(np.int32)
        for u,v in uv_int:
            cv2.circle(rgb, (u,v), 1, (0,255,0), -1)
        Image.fromarray(rgb).save(outdir / f"{stem}_points_on_rgb.png")

    # dump stats json
    (outdir / f"{stem}_stats.json").write_text(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()

