#!/usr/bin/env python3
import argparse, shutil, csv
from pathlib import Path
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def save_as_jpg(src: Path, dst: Path, quality: int = 92):
    """Load any image and save as JPEG to dst (RGB)."""
    im = Image.open(src).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    im.save(dst, format="JPEG", quality=quality, optimize=True)

def copy_pairs(src_root: Path, dst_root: Path, prefix: str, skip_existing: bool, quality: int, writer):
    """
    Copy all pairs from a toy root (images/*.jpg + lidar/*.npy) into dst_root,
    renaming to {prefix}_{stem}.jpg/.npy. If images are not .jpg, convert to .jpg.
    """
    simg = src_root / "images"
    slid = src_root / "lidar"
    dst_img = dst_root / "images"; dst_img.mkdir(parents=True, exist_ok=True)
    dst_lid = dst_root / "lidar";  dst_lid.mkdir(parents=True, exist_ok=True)

    n = 0
    # accept multiple image extensions, but emit .jpg in the destination
    image_files = sorted([p for p in simg.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])

    for img in image_files:
        stem = img.stem
        lid = slid / f"{stem}.npy"
        if not lid.exists():
            continue

        out_img = dst_img / f"{prefix}_{stem}.jpg"   # always .jpg
        out_lid = dst_lid / f"{prefix}_{stem}.npy"

        if skip_existing and out_img.exists() and out_lid.exists():
            # still log the mapping so pairs_merged.csv has a complete record
            writer.writerow([out_img.name, out_lid.name, str(img), str(lid), prefix])
            continue

        # image: copy or convert to .jpg
        if img.suffix.lower() == ".jpg" or img.suffix.lower() == ".jpeg":
            shutil.copy2(img, out_img)
        else:
            save_as_jpg(img, out_img, quality=quality)

        # lidar: copy .npy 1:1
        shutil.copy2(lid, out_lid)

        writer.writerow([out_img.name, out_lid.name, str(img), str(lid), prefix])
        n += 1
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="list of toy roots (each has images/ and lidar/)")
    ap.add_argument("--out_root", required=True,
                    help="destination MERGED toy root (should be the toy folder itself, e.g. datasets/nuscenes_toy)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="if set, do not overwrite existing pairs with the same merged name")
    ap.add_argument("--quality", type=int, default=92,
                    help="JPEG quality when converting .png -> .jpg")
    args = ap.parse_args()

    out = Path(args.out_root)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "lidar").mkdir(parents=True, exist_ok=True)
    csv_path = out / "pairs_merged.csv"

    # open CSV in append mode so you can keep growing the merged toy
    csv_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as fcsv:
        writer = csv.writer(fcsv)
        if not csv_exists:
            writer.writerow(["out_image", "out_lidar", "src_image", "src_lidar", "prefix"])

        total = 0
        for inp in args.inputs:
            src = Path(inp)
            if not (src / "images").exists() or not (src / "lidar").exists():
                print(f"[skip] {src} missing images/ or lidar/ (is it a toy root?)")
                continue
            prefix = src.name  # unique prefix from folder name
            n = copy_pairs(src, out, prefix, args.skip_existing, args.quality, writer)
            print(f"[merge] {src} -> {out}  (+{n} pairs)")
            total += n

    print(f"[done] merged total {total} pairs into {out}")
    print(f"[log] {csv_path}")

if __name__ == "__main__":
    main()

