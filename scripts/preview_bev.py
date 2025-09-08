#!/usr/bin/env python3
"""
Preview a single LiDAR frame as a BEV raster and (optionally) print stats.

Examples
--------
export PYTHONPATH=$PWD
LIDAR=~/data/.../points/xxx.bin

python scripts/preview_bev.py "$LIDAR" \
  --out /tmp/one_bev.png \
  --H 128 --W 128 \
  --meters_x 80 --meters_y 60 \
  --unit_scale 0.01 \
  --z_min -3 --z_max 3 \
  --fov_deg 360 \
  --blur_ksize 3 --close_ks 3 --dilate_ks 5 \
  --swap_xy --flip_x --flip_y \
  --weight_mode near --near_sigma 12 \
  --range_min 1.0 --range_max 60.0 \
  --stats
"""
import argparse, os, sys, math
from pathlib import Path
import numpy as np
from PIL import Image

# Optional OpenCV (for blur/dilate/close)
try:
    import cv2
    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

# Optional sklearn for SOR (statistical outlier removal)
try:
    from sklearn.neighbors import NearestNeighbors
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def load_points(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    ext = p.suffix.lower()
    if ext == ".npy":
        pts = np.load(p)
    elif ext == ".npz":
        arrs = np.load(p)
        # take the first array inside
        key = list(arrs.keys())[0]
        pts = arrs[key]
    elif ext == ".bin":
        raw = np.fromfile(p, dtype=np.float32)
        # try Nx4 or Nx3
        if raw.size % 4 == 0:
            pts = raw.reshape(-1, 4)
        elif raw.size % 3 == 0:
            pts = raw.reshape(-1, 3)
        else:
            # fallback: assume Nx4 and trim tail
            n4 = raw.size // 4
            pts = raw[: n4 * 4].reshape(-1, 4)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        # make safe empty
        return np.zeros((0, 3), dtype=np.float32)
    return pts


def apply_orientation(pts: np.ndarray, unit_scale=1.0,
                      swap_xy=False, flip_x=False, flip_y=False) -> np.ndarray:
    pts = pts.copy()
    if unit_scale != 1.0:
        pts[:, :3] *= float(unit_scale)
    if swap_xy:
        pts[:, [0, 1]] = pts[:, [1, 0]]
    if flip_x:
        pts[:, 0] = -pts[:, 0]
    if flip_y:
        pts[:, 1] = -pts[:, 1]
    return pts


def crop_points(
    pts,
    z_min=None, z_max=None,
    fov_deg=None,
    swap_xy=False, flip_x=False, flip_y=False,
    range_min=None, range_max=None,
):
    """
    Filters and orients raw points.
    Accepts Nx3 (x,y,z) or Nx4 (x,y,z,intensity) arrays.
    All filters are ANDed together:
      - finite coordinates
      - z band [z_min, z_max]
      - FOV in degrees (centered on +x), if < 360
      - radial range [range_min, range_max]
    Applies the same mask to intensity if present.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return np.zeros((0, 4 if (pts.ndim == 2 and pts.shape[1] > 3) else 3), np.float32)

    # split columns
    x = pts[:, 0].copy()
    y = pts[:, 1].copy()
    z = pts[:, 2].copy()
    i = pts[:, 3].copy() if pts.shape[1] > 3 else None

    # orientation fixes
    if swap_xy:
        x, y = y, x
    if flip_x:
        x = -x
    if flip_y:
        y = -y

    # combined mask
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    if z_min is not None and z_max is not None:
        m &= (z >= float(z_min)) & (z <= float(z_max))

    if fov_deg is not None and fov_deg < 360:
        ang = np.degrees(np.arctan2(y, x))
        half = float(fov_deg) / 2.0
        m &= (ang >= -half) & (ang <= half)

    if (range_min is not None) or (range_max is not None):
        r = np.sqrt(x*x + y*y)
        if range_min is not None:
            m &= (r >= float(range_min))
        if range_max is not None:
            m &= (r <= float(range_max))

    # apply mask
    x, y, z = x[m], y[m], z[m]
    if i is not None:
        i = i[m]

    out = np.c_[x, y, z, i] if i is not None else np.c_[x, y, z]
    return out.astype(np.float32)


def statistical_outlier_removal(pts: np.ndarray, k=16, std=2.0) -> np.ndarray:
    """
    Remove points whose mean nn-distance is > mean + std*stddev.
    Requires scikit-learn; silently returns pts if unavailable.
    """
    if not HAVE_SK or pts.shape[0] < max(4, k + 1):
        return pts
    xyz = pts[:, :3]
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(xyz)
    dists, _ = nn.kneighbors(xyz)
    # skip self-distance at col 0
    md = dists[:, 1:].mean(axis=1)
    mu, sigma = md.mean(), md.std()
    keep = md <= (mu + std * sigma)
    return pts[keep]


def rasterize_bev(pts: np.ndarray, H=128, W=128, meters_x=60.0, meters_y=30.0,
                  weight_mode="count", near_sigma=12.0) -> np.ndarray:
    """
    Build a BEV occupancy/weight map with chosen weighting.
    x in [0, meters_x], y in [-meters_y/2, +meters_y/2] maps to W x H.
    """
    bev = np.zeros((H, W), dtype=np.float32)
    if pts.size == 0:
        return bev

    xr, yr = float(meters_x), float(meters_y)
    x1, x2 = -1e-6, xr
    y1, y2 = -yr / 2.0, yr / 2.0

    x = pts[:, 0]
    y = pts[:, 1]
    keep = (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)
    if not keep.any():
        return bev

    x, y = x[keep], y[keep]

    # grid indices (flip y so top is +y)
    gx = ((x - x1) / (x2 - x1) * (W - 1)).astype(np.int32)
    gy = ((y - y1) / (y2 - y1) * (H - 1)).astype(np.int32)
    gy = (H - 1 - gy)

    if weight_mode == "intensity" and pts.shape[1] > 3:
        w = np.asarray(pts[keep, 3], dtype=np.float32)
        # normalize intensities to [0,1] per-frame
        if np.isfinite(w).any():
            wmin, wmax = float(np.nanmin(w)), float(np.nanmax(w))
            if wmax > wmin:
                w = (w - wmin) / (wmax - wmin)
            else:
                w = np.ones_like(w, dtype=np.float32)
        else:
            w = np.ones_like(w, dtype=np.float32)
    elif weight_mode == "near":
        r = np.sqrt(x * x + y * y)
        sigma = float(near_sigma)
        # gaussian weighting toward near
        w = np.exp(-(r * r) / (2.0 * sigma * sigma)).astype(np.float32)
    else:
        w = np.ones_like(x, dtype=np.float32)

    # accumulate (use np.add.at for duplicates)
    np.add.at(bev, (gy, gx), w)

    # log compress then normalize
    bev = np.log1p(bev)
    m = bev.max()
    if m > 0:
        bev = bev / m
    return bev


def postprocess_bev(bev: np.ndarray, blur_ksize=0, close_ks=0, dilate_ks=0) -> np.ndarray:
    """Optional smoothing/morphology (requires OpenCV); returns [0,1] float."""
    out = bev.astype(np.float32, copy=True)

    if HAVE_CV2:
        if blur_ksize and blur_ksize >= 3 and (blur_ksize % 2 == 1):
            out = cv2.GaussianBlur(out, (blur_ksize, blur_ksize), 0)

        if close_ks and close_ks >= 3 and (close_ks % 2 == 1):
            k = np.ones((close_ks, close_ks), np.uint8)
            out = cv2.morphologyEx((out * 255).astype(np.uint8), cv2.MORPH_CLOSE, k)
            out = out.astype(np.float32) / 255.0

        if dilate_ks and dilate_ks >= 3 and (dilate_ks % 2 == 1):
            k = np.ones((dilate_ks, dilate_ks), np.uint8)
            out = cv2.dilate((out * 255).astype(np.uint8), k, iterations=1)
            out = out.astype(np.float32) / 255.0

    # re-normalize in case morphology expanded range slightly
    m = out.max()
    if m > 0:
        out = out / m
    return out


def save_png(bev: np.ndarray, out_path: str):
    # contrast stretch for visibility
    lo = np.percentile(bev, 1.0)
    hi = np.percentile(bev, 99.0)
    if hi <= lo:
        hi = lo + 1e-6
    y = np.clip((bev - lo) / (hi - lo), 0.0, 1.0)
    img = (y * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(img, mode="L").save(out_path)


def print_stats(bev: np.ndarray, pts_count: int):
    nz = int((bev > 0).sum())
    H, W = bev.shape
    occ = nz / float(H * W)
    print(f"BEV stats: shape=({H}, {W}), min={bev.min():.4f}, max={bev.max():.4f}, mean={bev.mean():.6f}, "
          f"nonzero={nz} ({occ*100:.3f}% of grid), points_in_window={pts_count}")

    # histogram (coarse)
    hist, edges = np.histogram(bev, bins=[0.0, 1e-6, 0.01, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
    bins_str = ", ".join([f"[{edges[i]:.3f},{edges[i+1]:.3f}):{hist[i]}" for i in range(len(hist))])
    print(f"BEV histogram: {bins_str}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lidar_bin", help="path to one LiDAR frame (.bin/.npy/.npz)")
    ap.add_argument("--out", required=True, help="output image path (png/jpg)")
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

    # post-processing
    ap.add_argument("--blur_ksize", type=int, default=0, help="odd >=3 or 0")
    ap.add_argument("--close_ks", type=int, default=0, help="odd >=3 or 0")
    ap.add_argument("--dilate_ks", type=int, default=0, help="odd >=3 or 0")

    # filtering / weighting
    ap.add_argument("--range_min", type=float, default=None)
    ap.add_argument("--range_max", type=float, default=None)
    ap.add_argument("--weight_mode", choices=["count", "intensity", "near"], default="count")
    ap.add_argument("--near_sigma", type=float, default=12.0)
    ap.add_argument("--sor_k", type=int, default=0, help="neighbors for statistical outlier removal (requires sklearn). 0=off")
    ap.add_argument("--sor_std", type=float, default=2.0)

    # NEW: print detailed stats
    ap.add_argument("--stats", action="store_true", help="Print detailed BEV statistics to stdout")

    args = ap.parse_args()

    pts = load_points(args.lidar_bin)
    pts = apply_orientation(pts, unit_scale=args.unit_scale,
                            swap_xy=args.swap_xy, flip_x=args.flip_x, flip_y=args.flip_y)
    pts = crop_points(pts, args.z_min, args.z_max, args.fov_deg,
                      range_min=args.range_min, range_max=args.range_max)

    if args.sor_k and HAVE_SK:
        pts = statistical_outlier_removal(pts, k=args.sor_k, std=args.sor_std)
    elif args.sor_k and not HAVE_SK:
        print("[preview_bev] sklearn not found; skipping SOR.", file=sys.stderr)

    bev = rasterize_bev(
        pts, H=args.H, W=args.W,
        meters_x=args.meters_x, meters_y=args.meters_y,
        weight_mode=args.weight_mode, near_sigma=args.near_sigma
    )
    bev = postprocess_bev(bev, blur_ksize=args.blur_ksize, close_ks=args.close_ks, dilate_ks=args.dilate_ks)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    save_png(bev, args.out)
    print(f"Wrote {args.out}")

    if args.stats:
        # count points that actually contributed inside the raster window
        xr, yr = float(args.meters_x), float(args.meters_y)
        in_win = ((pts[:, 0] >= -1e-6) & (pts[:, 0] <= xr) &
                  (pts[:, 1] >= -yr/2.0) & (pts[:, 1] <= yr/2.0)).sum() if pts.size else 0
        print_stats(bev, int(in_win))
    else:
        # legacy short line
        print(f"BEV stats: shape=({args.H}, {args.W}), min={bev.min():.4f}, max={bev.max():.4f}, mean={bev.mean():.4f}")


if __name__ == "__main__":
    main()

