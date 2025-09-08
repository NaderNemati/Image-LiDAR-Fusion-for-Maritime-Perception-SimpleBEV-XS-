# scripts/make_pohang_pairs.py
#!/usr/bin/env python3
import argparse, os, re, sys, shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    import cv2
    _HAVE_CV2 = True
except Exception:
    _HAVE_CV2 = False

def parse_ts(stem: str):
    """Extract an integer timestamp from a filename stem (last run of digits)."""
    m = re.findall(r"\d+", stem)
    return int(m[-1]) if m else None

def load_points_xyz(path: Path):
    """Load LiDAR .bin (float32 x,y,z,intensity) or .npy/.npz -> Nx3 float32."""
    sfx = path.suffix.lower()
    if sfx == ".bin":
        raw = np.fromfile(str(path), dtype=np.float32)
        if raw.size % 4 != 0:  # unexpected
            return None
        pts = raw.reshape(-1, 4)[:, :3].astype(np.float32)  # keep XYZ
        return pts
    if sfx == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3].astype(np.float32)
        return None
    if sfx == ".npz":
        try:
            data = np.load(path)
            for k in data.files:
                arr = np.asarray(data[k], dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return arr[:, :3].astype(np.float32)
        except Exception:
            return None
    return None  # unsupported

def read_image(path: Path):
    """Read image as BGR uint8 (cv2) or RGB (PIL fallback)."""
    if _HAVE_CV2:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        return im  # BGR uint8 or 16-bit -> we will convert later
    else:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        return np.array(im)[:, :, ::-1]  # RGB->BGR to unify

def enhance_image_bgr(im, clahe=False, p_low=1.0, p_high=99.0):
    """Optional CLAHE + percentile contrast stretch in BGR space."""
    if im is None:
        return None
    if im.dtype != np.uint8:
        im = np.clip(im, 0, 255).astype(np.uint8)

    out = im.copy()
    if clahe:
        if out.ndim == 3 and out.shape[2] == 3:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            L, A, B = cv2.split(lab)
            cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            L2 = cla.apply(L)
            lab2 = cv2.merge([L2, A, B])
            out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        else:
            cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            out = cla.apply(out)

    # Contrast stretch per-channel
    for c in range(out.shape[2] if out.ndim == 3 else 1):
        ch = out[..., c] if out.ndim == 3 else out
        lo = np.percentile(ch, p_low); hi = np.percentile(ch, p_high)
        if hi > lo:
            ch = (ch.astype(np.float32) - lo) * (255.0 / (hi - lo))
            ch = np.clip(ch, 0, 255).astype(np.uint8)
            if out.ndim == 3:
                out[..., c] = ch
            else:
                out = ch
    return out

def next_index(images_dir: Path, lid_dir: Path):
    """Find the next 6-digit index to write, based on existing files."""
    def idxs(p: Path):
        vals = []
        for f in p.glob("*"):
            m = re.match(r"(\d+)", f.stem)
            if m:
                vals.append(int(m.group(1)))
        return vals
    existing = idxs(images_dir) + idxs(lid_dir)
    return (max(existing) + 1) if existing else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True, help="root containing camera and LiDAR subfolders")
    ap.add_argument("--out_root", required=True, help="repo datasets/ directory")
    ap.add_argument("--toy_name", default="nuscenes_toy")
    ap.add_argument("--cam_glob", default="stereo/right_images/*.png")
    ap.add_argument("--lidar_glob", default="lidar/lidar_front/points/*.bin")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth camera frame")
    ap.add_argument("--max_pairs", type=int, default=100000)
    ap.add_argument("--write_ext", choices=["jpg","png"], default="jpg")
    ap.add_argument("--quality", type=int, default=92, help="JPEG quality")
    ap.add_argument("--clahe", action="store_true", help="apply CLAHE before saving")
    ap.add_argument("--p_low", type=float, default=1.0, help="low percentile for contrast stretch")
    ap.add_argument("--p_high", type=float, default=99.0, help="high percentile for contrast stretch")
    ap.add_argument("--append", action="store_true", help="append to existing dataset (don’t clear)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    src_root = Path(args.src_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()
    toy = out_root / args.toy_name
    img_out = toy / "images"
    lid_out = toy / "lidar"

    if not args.append:
        # fresh dataset
        if toy.exists():
            shutil.rmtree(toy)
    img_out.mkdir(parents=True, exist_ok=True)
    lid_out.mkdir(parents=True, exist_ok=True)

    # scan
    cams = sorted(src_root.glob(args.cam_glob))
    lids = sorted(src_root.glob(args.lidar_glob))
    print(f"[scan] found {len(cams)} images and {len(lids)} lidar files under {src_root}")
    if len(cams) == 0 or len(lids) == 0:
        print("[!] Nothing to do.")
        return

    # subsample cameras by stride
    cams = cams[::max(1, args.stride)]

    # timestamps for lids
    lid_ts = np.array([parse_ts(p.stem) if parse_ts(p.stem) is not None else -1 for p in lids])

    # starting index
    idx = next_index(img_out, lid_out)

    # dry-run preview
    if args.dry_run:
        print(f"[pair] would write up to {min(len(cams), args.max_pairs)} pairs to {toy} (starting index={idx})")
        print("[dry_run] stopping before writing any files.")
        return

    wrote = 0
    for cam in tqdm(cams[:args.max_pairs], desc="Pairing & writing"):
        # pair by nearest timestamp (fallback: rolling index)
        tsim = parse_ts(cam.stem)
        if tsim is None or (lid_ts < 0).all():
            j = min(wrote, len(lids)-1)
        else:
            j = int(np.argmin(np.abs(lid_ts - tsim)))
        lid = lids[j]

        # read / validate LiDAR
        pts = load_points_xyz(lid)
        if pts is None or pts.shape[0] == 0:
            continue

        # read & enhance image
        im_bgr = read_image(cam)
        if im_bgr is None:
            continue
        im_bgr = enhance_image_bgr(im_bgr, clahe=args.clahe, p_low=args.p_low, p_high=args.p_high)

        # write outputs
        stem = f"{idx:06d}"
        if args.write_ext == "jpg":
            # cv2 expects BGR
            ok = cv2.imencode(".jpg", im_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)])[0] if _HAVE_CV2 else True
            if _HAVE_CV2:
                cv2.imwrite(str(img_out / f"{stem}.jpg"), im_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.quality)])
            else:
                from PIL import Image
                Image.fromarray(im_bgr[:, :, ::-1]).save(img_out / f"{stem}.jpg", quality=args.quality)
        else:
            if _HAVE_CV2:
                cv2.imwrite(str(img_out / f"{stem}.png"), im_bgr)
            else:
                from PIL import Image
                Image.fromarray(im_bgr[:, :, ::-1]).save(img_out / f"{stem}.png")

        np.save(lid_out / f"{stem}.npy", pts.astype(np.float32))
        idx += 1
        wrote += 1

    print(f"Done. Wrote {wrote} pairs to {img_out} and {lid_out}")

if __name__ == "__main__":
    main()

