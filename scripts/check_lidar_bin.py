from pathlib import Path
import numpy as np, sys, os

def stats(name, pts):
    if pts.size == 0:
        print(f"{name}: EMPTY")
        return
    print(f"{name}: shape={pts.shape}")
    q95 = np.percentile(np.abs(pts), 95)
    print("  95%% |x,y,z|:", q95)
    print("  min/max x:", float(pts[:,0].min()), float(pts[:,0].max()))
    print("  min/max y:", float(pts[:,1].min()), float(pts[:,1].max()))
    print("  min/max z:", float(pts[:,2].min()), float(pts[:,2].max()))

def main():
    p = Path(sys.argv[1]).expanduser()
    bs = p.stat().st_size
    print("file:", p)
    print("size:", bs, "bytes")

    # float32
    try:
        raw32 = np.fromfile(str(p), dtype=np.float32)
        print("float32 count:", raw32.size)
        if raw32.size % 4 == 0:
            pts = raw32.reshape(-1,4)[:, :3]
            stats("float32 Nx4 (m?)", pts)
    except Exception as e:
        print("float32 read error:", e)

    # float64
    try:
        raw64 = np.fromfile(str(p), dtype=np.float64)
        print("float64 count:", raw64.size)
        if raw64.size % 4 == 0:
            pts = raw64.reshape(-1,4)[:, :3].astype(np.float32)
            stats("float64 Nx4 (m?)", pts)
    except Exception as e:
        print("float64 read error:", e)

    # int16 (mm/cm)
    try:
        raw16 = np.fromfile(str(p), dtype=np.int16)
        print("int16 count:", raw16.size)
        if raw16.size % 4 == 0:
            cand = raw16.reshape(-1,4)[:, :3].astype(np.float32)
            stats("int16 Nx4 scaled 0.001 (mm->m)", cand*0.001)
            stats("int16 Nx4 scaled 0.01  (cm->m)", cand*0.01)
    except Exception as e:
        print("int16 read error:", e)

    # int32 (mm)
    try:
        raw32i = np.fromfile(str(p), dtype=np.int32)
        print("int32 count:", raw32i.size)
        if raw32i.size % 4 == 0:
            cand = raw32i.reshape(-1,4)[:, :3].astype(np.float32)
            stats("int32 Nx4 scaled 0.001 (mm->m)", cand*0.001)
    except Exception as e:
        print("int32 read error:", e)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/check_lidar_bin.py <path/to/file.bin>")
        sys.exit(1)
    main()
