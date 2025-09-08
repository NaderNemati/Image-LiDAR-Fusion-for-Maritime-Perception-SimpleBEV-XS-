#!/usr/bin/env python3
import argparse, numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True, help="KITTI raw root containing date_drive sequences")
    ap.add_argument("--cam_glob", default="**/image_2/data/*.png")   # left color
    ap.add_argument("--lid_glob", default="**/velodyne_points/data/*.bin")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--toy_name", default="nuscenes_toy")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out_root)/args.toy_name
    imgs = out/"images"; lids = out/"lidar"
    imgs.mkdir(parents=True, exist_ok=True); lids.mkdir(parents=True, exist_ok=True)

    cam_files = sorted(Path(args.kitti_root).glob(args.cam_glob))
    lid_files = sorted(Path(args.kitti_root).glob(args.lid_glob))
    if not cam_files or not lid_files:
        print("[!] No KITTI files found"); return

    def load_bin_xyz(fp: Path):
        arr = np.fromfile(str(fp), dtype=np.float32).reshape(-1,4)
        return arr[:, :3]

    used = 0
    for i, im in enumerate(tqdm(cam_files[::args.stride], desc="Export")):
        if i >= len(lid_files): break
        lp = lid_files[i]
        xyz = load_bin_xyz(lp)
        Image.open(im).convert("RGB").save(imgs/f"{im.stem}.jpg")
        np.save(lids/f"{im.stem}.npy", xyz.astype(np.float32))
        used += 1

    print("Wrote", used, "pairs to", out)

if __name__ == "__main__":
    main()

