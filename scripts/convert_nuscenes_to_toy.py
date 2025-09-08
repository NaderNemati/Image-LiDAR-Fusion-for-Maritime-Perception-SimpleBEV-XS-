#!/usr/bin/env python3
import argparse, os, numpy as np
from pathlib import Path
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nusc", required=True, help="path to downloaded nuScenes root (v1.0-mini or full)")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--toy_name", default="nuscenes_toy")
    ap.add_argument("--camera", default="CAM_FRONT", help="which camera to export")
    ap.add_argument("--limit", type=int, default=1_000_000)
    args = ap.parse_args()

    out = Path(args.out_root)/args.toy_name
    imgs = out/"images"; lids = out/"lidar"
    imgs.mkdir(parents=True, exist_ok=True); lids.mkdir(parents=True, exist_ok=True)

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import LidarPointCloud
    from PIL import Image

    nusc = NuScenes(version="v1.0-mini" if "mini" in args.nusc else "v1.0-trainval", dataroot=args.nusc, verbose=True)

    count=0
    for sample in tqdm(nusc.sample, desc="Export"):
        if count >= args.limit: break
        cam_token = sample['data'].get(args.camera, None)
        lidar_token = sample['data'].get('LIDAR_TOP', None)
        if cam_token is None or lidar_token is None: continue

        sd_cam = nusc.get('sample_data', cam_token)
        sd_lid = nusc.get('sample_data', lidar_token)
        im_path = Path(nusc.get_sample_data_path(cam_token))
        lid_path = Path(nusc.get_sample_data_path(lidar_token))

        # image
        try:
            im = Image.open(im_path).convert("RGB")
            im.save(imgs/f"{sd_cam['token']}.jpg")
        except Exception:
            continue

        # lidar (to Nx3 npy)
        try:
            pc = LidarPointCloud.from_file(str(lid_path))
            xyz = pc.points[:3, :].T.astype(np.float32)
            np.save(lids/f"{sd_cam['token']}.npy", xyz)
        except Exception:
            continue

        count += 1

    print("Wrote", count, "pairs to", out)

if __name__ == "__main__":
    main()

